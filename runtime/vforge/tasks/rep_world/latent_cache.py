"""Read committed VideoRAE3D shards at the frozen-codec training boundary.

This task-owned format stores packed, normalized FP32 latents, not trainable
conditioner features. Native still owns sampling, worker RNG and data cursors.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vforge.tasks.rep_world.data import VideoClipDataset, frame_indices, read_video_clip
from vforge.utils.training import resolve_dtype

SHARD_SCHEMA = "rep_world.videorae3d.packed_fp32.shards.v1"
CONSECUTIVE_SCHEMA = "solarwm.videorae3d.consecutive160.packed_fp32.shards.v1"


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"RepWorld cache metadata is missing or invalid: {path}") from error


class CachedVideoLatents(Dataset):
    """Exact source-manifest order, with one original clip RNG draw per read.

    Incomplete caches are rejected. Shards use bounded lazy mmap handles;
    per-row hashes detect corruption without reading 640 MiB for each sample
    selected by a shuffled loader. Returned tensors own writable CPU storage.
    """

    def __init__(self, root, manifest, *, config, precision, expected_fingerprint=""):
        self.root = Path(root).resolve()
        contract = _json(self.root / "contract.json")
        fingerprint = contract.get("fingerprint")
        unsigned = {key: value for key, value in contract.items() if key != "fingerprint"}
        actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        if fingerprint != actual or (expected_fingerprint and fingerprint != expected_fingerprint):
            raise ValueError("RepWorld cache fingerprint mismatch")
        if contract.get("schema") != SHARD_SCHEMA:
            raise ValueError("Unsupported RepWorld latent shard schema")
        complete = _json(self.root / "COMPLETE.json")
        if complete.get("status") != "complete" or complete.get("fingerprint") != fingerprint:
            raise ValueError("RepWorld cache is incomplete or has a different fingerprint")
        frames = config.states * config.geometry.rgb_frames
        original = VideoClipDataset(
            manifest,
            frames=frames,
            frame_interval=config.frame_interval,
            random_clip=config.random_clip,
        )
        if (
            contract.get("source_manifest_sha256") != original.resume_contract["manifest_digest"]
            or contract.get("samples") != len(original)
            or complete.get("samples") != len(original)
        ):
            raise ValueError("Cache does not match the exact ordered data.train_manifest")
        geometry = config.geometry
        self.shape = (
            geometry.planes * geometry.channels,
            config.states,
            geometry.grid,
            geometry.grid,
        )
        if (
            (geometry.family, geometry.variant) != ("videorae", "3d")
            or contract.get("codec", {}).get("family") != geometry.family
            or contract.get("codec", {}).get("variant") != geometry.variant
            or contract.get("latent_shape") != list(self.shape)
            or contract.get("latent_dtype") != "float32"
            or contract.get("storage_order") != "C"
            or contract.get("stored_key") != "latents"
            or contract.get("channel_index") != "plane * 64 + channel"
            or contract.get("states_per_clip") != config.states
            or contract.get("rgb_shape") != [3, frames, 256, 256]
            or contract.get("frame_interval") != config.frame_interval
            or config.frame_interval != 1
            or contract.get("frames") != list(range(frames))
            or contract.get("resize") != [256, 256]
        ):
            raise ValueError(
                "RepWorld cache geometry, packing or frame selection does not match training"
            )
        # v1 artifacts were explicitly encoded with BF16 weights/autocast. Do
        # not silently replace an FP32 online encoder with those cached values.
        if resolve_dtype(precision) != torch.bfloat16 or not contract.get(
            "precision", ""
        ).startswith("bf16 "):
            raise ValueError("This RepWorld cache requires the BF16 codec precision contract")
        for key in ("checkpoint", "statistics"):
            declared = config.codec.get(key) or ""
            asset = contract.get("assets", {}).get(key)
            if (
                not declared
                or not isinstance(asset, dict)
                or _digest(declared) != asset.get("sha256")
            ):
                raise ValueError(f"RepWorld cached codec {key} does not match configured asset")
        self.fingerprint = fingerprint
        self.records = original.records
        self.frames, self.random_clip = frames, config.random_clip
        self.shard_size = contract.get("shard_size")
        if type(self.shard_size) is not int or self.shard_size < 1:
            raise ValueError("Invalid RepWorld cache shard_size")
        count = (len(self.records) + self.shard_size - 1) // self.shard_size
        descriptors = complete.get("shards", [])
        if len(descriptors) != count:
            raise ValueError("RepWorld cache completion does not cover every shard")
        for sid, descriptor in enumerate(descriptors):
            size = min(self.shard_size, len(self.records) - sid * self.shard_size)
            if (
                descriptor.get("shard") != sid
                or descriptor.get("count") != size
                or descriptor.get("fingerprint") != fingerprint
                or descriptor.get("file") != f"{sid:06d}.npy"
                or descriptor.get("verified_readback") is not True
                or not (self.root / "shards" / f"{sid:06d}.npy").is_file()
                or not (self.root / "shards" / f"{sid:06d}.json").is_file()
            ):
                raise ValueError(f"RepWorld cache shard {sid} is not committed")
        self.descriptors = descriptors
        index = self.root / "index.jsonl"
        if _digest(index) != complete.get("index_sha256"):
            raise ValueError("RepWorld cache index digest mismatch")
        self.row_hashes = []
        with index.open() as handle:
            for offset, line in enumerate(handle):
                row = json.loads(line)
                sid, position = divmod(offset, self.shard_size)
                if (
                    offset >= len(self.records)
                    or row.get("index") != offset
                    or row.get("sample_id") != self.records[offset]["sample_id"]
                    or row.get("latent_path") != f"shards/{sid:06d}.npy"
                    or row.get("row") != position
                    or not isinstance(row.get("latent_sha256"), str)
                ):
                    raise ValueError(f"RepWorld cache sample order mismatch at index {offset}")
                self.row_hashes.append(row["latent_sha256"])
        if len(self.row_hashes) != len(self.records):
            raise ValueError("RepWorld cache index is incomplete")
        self.resume_contract = {
            **original.resume_contract,
            "dataset": "rep_world.latent_shards.v1",
            "cache_dir": str(self.root),
            "cache_fingerprint": fingerprint,
        }
        self.include_rgb = config.conditioner is not None and config.conditioner.input == "rgb"
        if self.include_rgb:
            self.resume_contract = {
                **self.resume_contract,
                "dataset": "rep_world.latent_shards_with_rgb.v1",
                "rgb_fixed_window": True,
            }
        self._shards = OrderedDict()

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "_shards": OrderedDict()}

    def _open_shard(self, sid):
        if sid in self._shards:
            self._shards.move_to_end(sid)
            return self._shards[sid]
        metadata = _json(self.root / "shards" / f"{sid:06d}.json")
        expected = self.descriptors[sid]
        if any(
            metadata.get(key) != expected[key]
            for key in ("shard", "count", "file", "fingerprint", "sha256")
        ):
            raise ValueError(f"RepWorld cache shard {sid} metadata mismatch")
        records = metadata.get("records", [])
        if len(records) != expected["count"]:
            raise ValueError(f"RepWorld cache shard {sid} row count mismatch")
        for position, row in enumerate(records):
            index = sid * self.shard_size + position
            if (
                row.get("index") != index
                or row.get("row") != position
                or row.get("sample_id") != self.records[index]["sample_id"]
                or row.get("rgb_path") != self.records[index]["path"]
                or row.get("latent_sha256") != self.row_hashes[index]
                or row.get("source_frames") != self.frames
                or row.get("frame_start") != 0
                or row.get("frame_interval") != 1
            ):
                raise ValueError(f"RepWorld cache shard {sid} source window/order mismatch")
        array = np.load(self.root / "shards" / expected["file"], mmap_mode="r", allow_pickle=False)
        if (
            array.dtype != np.float32
            or array.shape != (expected["count"], *self.shape)
            or not array.flags.c_contiguous
        ):
            raise ValueError(f"RepWorld cache shard {sid} dtype/shape/layout mismatch")
        self._shards[sid] = array
        if len(self._shards) > 16:
            _, previous = self._shards.popitem(last=False)
            previous._mmap.close()
        return array

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        record = self.records[index]
        try:
            sid, position = divmod(index, self.shard_size)
            value = self._open_shard(sid)[position].copy()
            if hashlib.sha256(memoryview(value).cast("B")).hexdigest() != self.row_hashes[index]:
                raise ValueError("latent checksum mismatch")
            if not np.isfinite(value).all():
                raise ValueError("nonfinite cached latent")
            result = {"sample_id": record["sample_id"], "latents": torch.from_numpy(value)}
            if self.include_rgb:
                result["rgb"] = read_video_clip(
                    record["path"], self.frames, 1, random_clip=self.random_clip, exact_length=True
                )
            else:
                # Even length==required consumes one draw when random_clip=True.
                frame_indices(self.frames, self.frames, 1, random_clip=self.random_clip)
            return result
        except Exception as error:
            raise RuntimeError(f"Cached sample {record['sample_id']}: {error}") from error


class CachedConsecutiveVideoLatents(Dataset):
    """Fixed-window manifest over committed VideoRAE shards from multiple owners.

    The selected window and its shard coordinate are explicit in the manifest.
    Cache metadata and row checksums are validated before a latent is returned;
    no source video or window is substituted on a read error.
    """

    def __init__(self, root, manifest, *, config, precision):
        self.root = Path(root).resolve()
        manifest = Path(manifest).resolve()
        split = _json(manifest.parent / "split_contract.json")
        if split.get("schema") != "rep_world.consecutive160.clip_split.v1":
            raise ValueError("Missing multi-source clip split contract")
        identity = split.get("manifests", {}).get(manifest.name)
        if not identity or identity.get("sha256") != _digest(manifest):
            raise ValueError("Multi-source training manifest digest mismatch")
        geometry = config.geometry
        self.state_view = getattr(config, "cached_state_view", "full")
        source_states = 10 if self.state_view == "random_9_from_10" else config.states
        self.shape = (
            geometry.planes * geometry.channels,
            source_states,
            geometry.grid,
            geometry.grid,
        )
        frames = source_states * geometry.rgb_frames
        if (
            (geometry.family, geometry.variant) != ("videorae", "3d")
            or frames != 160
            or config.frame_interval != 1
            or config.random_clip
            or resolve_dtype(precision) != torch.bfloat16
        ):
            raise ValueError(
                "Consecutive160 cache requires fixed BF16 VideoRAE3D 160-frame windows"
            )
        if Path(split.get("cache_root", "")).resolve() != self.root:
            raise ValueError("Clip split points to a different cache root")
        self.owners = {}
        self.descriptors = {}
        asset_digests = {}
        for owner, expected in split.get("owners", {}).items():
            directory = self.root / owner
            contract = _json(directory / "contract.json")
            unsigned = {key: value for key, value in contract.items() if key != "fingerprint"}
            fingerprint = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
            complete = _json(directory / "COMPLETE.json")
            if (
                contract.get("schema") != CONSECUTIVE_SCHEMA
                or contract.get("dataset") != owner
                or contract.get("fingerprint") != fingerprint
                or expected.get("fingerprint") != fingerprint
                or complete.get("status") != "complete"
                or complete.get("fingerprint") != fingerprint
                or complete.get("samples") != contract.get("samples")
                or expected.get("samples") != contract.get("samples")
                or complete.get("index_sha256") != expected.get("index_sha256")
                or _digest(directory / "index.jsonl") != expected.get("index_sha256")
                or contract.get("latent_shape") != list(self.shape)
                or contract.get("latent_dtype") != "float32"
                or contract.get("channel_index") != "plane * 64 + channel"
                or contract.get("window_frames") != frames
                or contract.get("frame_interval") != 1
                or contract.get("rgb_shape") != [3, frames, 256, 256]
                or not contract.get("precision", "").startswith("bf16 ")
            ):
                raise ValueError(f"Consecutive160 cache contract mismatch: {owner}")
            for key in ("checkpoint", "statistics"):
                declared = config.codec.get(key) or ""
                if declared and declared not in asset_digests:
                    asset_digests[declared] = _digest(declared)
                if not declared or asset_digests[declared] != contract.get("assets", {}).get(
                    key, {}
                ).get("sha256"):
                    raise ValueError(f"Consecutive160 codec {key} mismatch: {owner}")
            shard_size = contract.get("shard_size")
            descriptors = complete.get("shards", [])
            if (
                type(shard_size) is not int
                or shard_size < 1
                or len(descriptors) != (contract["samples"] + shard_size - 1) // shard_size
            ):
                raise ValueError(f"Invalid shard coverage: {owner}")
            for sid, descriptor in enumerate(descriptors):
                size = min(shard_size, contract["samples"] - sid * shard_size)
                if (
                    descriptor.get("shard") != sid
                    or descriptor.get("count") != size
                    or descriptor.get("fingerprint") != fingerprint
                    or descriptor.get("verified_readback") is not True
                    or not (directory / "shards" / f"{sid:06d}.npy").is_file()
                    or not (directory / "shards" / f"{sid:06d}.json").is_file()
                ):
                    raise ValueError(f"Uncommitted shard {owner}/{sid}")
            self.owners[owner] = (directory, shard_size, contract["samples"])
            self.descriptors[owner] = descriptors
        if not self.owners:
            raise ValueError("No committed cache owners")
        self.records = []
        self.by_shard = {}
        seen = set()
        with manifest.open() as handle:
            for line in handle:
                record = json.loads(line)
                owner = record.get("dataset")
                if owner not in self.owners:
                    raise ValueError(f"Unknown cache owner: {owner}")
                _, shard_size, count = self.owners[owner]
                offset = record.get("cache_index")
                start = record.get("start_frame")
                if (
                    type(offset) is not int
                    or not 0 <= offset < count
                    or type(start) is not int
                    or start < 0
                    or record.get("end_frame_exclusive") != start + frames
                    or not isinstance(record.get("latent_sha256"), str)
                    or len(record["latent_sha256"]) != 64
                    or not isinstance(record.get("rgb_path"), str)
                    or not record["rgb_path"]
                ):
                    raise ValueError(f"Invalid cached window: {record.get('sample_id')}")
                sample_id = record.get("sample_id")
                if not isinstance(sample_id, str) or sample_id in seen:
                    raise ValueError(f"Duplicate/invalid cached sample ID: {sample_id}")
                seen.add(sample_id)
                sid, row = divmod(offset, shard_size)
                entry = (
                    sample_id,
                    owner,
                    sid,
                    row,
                    offset,
                    record["rgb_path"],
                    record["start_frame"],
                    record["latent_sha256"],
                )
                self.by_shard.setdefault((owner, sid), []).append(entry)
                self.records.append(entry)
        if len(self.records) != identity.get("samples") or not self.records:
            raise ValueError("Multi-source manifest count mismatch")
        self.include_rgb = config.conditioner is not None and config.conditioner.input == "rgb"
        self.resume_contract = {
            "dataset": "rep_world.consecutive160.latent_shards.v1",
            "manifest": str(manifest),
            "manifest_digest": identity["sha256"],
            "split_contract_sha256": _digest(manifest.parent / "split_contract.json"),
            "cache_dir": str(self.root),
            "owner_fingerprints": {
                owner: value["fingerprint"] for owner, value in split["owners"].items()
            },
            "frames": frames,
            "frame_interval": 1,
            "random_clip": False,
            "resize": [256, 256],
            "include_rgb": self.include_rgb,
        }
        if self.state_view != "full":
            self.resume_contract.update(
                model_states=config.states,
                cached_state_view=self.state_view,
            )
        self._shards = OrderedDict()

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "_shards": OrderedDict()}

    def _open_shard(self, owner, sid):
        key = (owner, sid)
        if key in self._shards:
            self._shards.move_to_end(key)
            return self._shards[key]
        directory = self.owners[owner][0]
        descriptor = self.descriptors[owner][sid]
        metadata = _json(directory / "shards" / f"{sid:06d}.json")
        if any(
            metadata.get(field) != descriptor[field]
            for field in ("shard", "count", "fingerprint", "sha256")
        ):
            raise ValueError(f"Shard metadata mismatch: {owner}/{sid}")
        rows = metadata.get("records", [])
        if len(rows) != descriptor["count"]:
            raise ValueError(f"Shard row count mismatch: {owner}/{sid}")
        for sample_id, _, _, row, offset, path, start, checksum in self.by_shard[key]:
            item = rows[row]
            if (
                item.get("index") != offset
                or item.get("row") != row
                or item.get("sample_id") != sample_id
                or item.get("rgb_path") != path
                or item.get("frame_start") != start
                or item.get("frame_end_exclusive") != start + 160
                or item.get("latent_sha256") != checksum
            ):
                raise ValueError(f"Shard source window mismatch: {sample_id}")
        array = np.load(directory / "shards" / f"{sid:06d}.npy", mmap_mode="r", allow_pickle=False)
        if (
            array.dtype != np.float32
            or array.shape != (descriptor["count"], *self.shape)
            or not array.flags.c_contiguous
        ):
            raise ValueError(f"Shard array layout mismatch: {owner}/{sid}")
        self._shards[key] = array
        if len(self._shards) > 16:
            _, previous = self._shards.popitem(last=False)
            previous._mmap.close()
        return array

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample_id, owner, sid, row, _, path, start, checksum = self.records[index]
        try:
            value = self._open_shard(owner, sid)[row].copy()
            if hashlib.sha256(memoryview(value).cast("B")).hexdigest() != checksum:
                raise ValueError("latent checksum mismatch")
            if not np.isfinite(value).all():
                raise ValueError("nonfinite cached latent")
            result = {"sample_id": sample_id, "latents": torch.from_numpy(value)}
            if self.include_rgb:
                result["rgb"] = read_video_clip(path, 160, 1, random_clip=False, start_frame=start)
            return result
        except Exception as error:
            raise RuntimeError(f"Cached sample {sample_id}: {error}") from error
