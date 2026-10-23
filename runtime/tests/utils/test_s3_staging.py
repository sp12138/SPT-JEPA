from __future__ import annotations

import shutil
import sys
from hashlib import sha256
from types import ModuleType

import pytest

from vforge.utils.s3_staging import (
    S3MountConfig,
    S3Source,
    StorageConfig,
    activate_storage_config,
    build_storage_config,
    download_tree_crt,
    resolve_s3_source,
    restore_storage_config,
    upload_tree_crt,
)


class FakeTransferConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_boto3(monkeypatch, client) -> None:
    boto3 = ModuleType("boto3")
    boto3.client = client
    transfer = ModuleType("boto3.s3.transfer")
    transfer.TransferConfig = FakeTransferConfig
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "boto3.s3", ModuleType("boto3.s3"))
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", transfer)


def test_resolve_s3_source_maps_mountpoint_path(tmp_path) -> None:
    mount = tmp_path / "mount"
    checkpoint = mount / "checkpoints" / "00002.dcp"
    checkpoint.mkdir(parents=True)

    token = activate_storage_config(
        StorageConfig(s3_mounts=(S3MountConfig(str(mount), "bucket", "region"),))
    )
    try:
        resolved = resolve_s3_source(str(checkpoint))
    finally:
        restore_storage_config(token)

    assert resolved == S3Source(
        bucket="bucket",
        key_prefix="checkpoints/00002.dcp",
        region="region",
    )


def test_resolve_s3_source_preserves_oss_routing_options(tmp_path) -> None:
    mount = tmp_path / "mount"
    checkpoint = mount / "MiniMax-H3" / "transformer"
    checkpoint.mkdir(parents=True)
    config = StorageConfig(
        s3_mounts=(
            S3MountConfig(
                mount_path=str(mount),
                bucket="models",
                region="oss-region",
                key_prefix="shared/models",
                endpoint_url="https://oss.example.com",
                addressing_style="virtual",
            ),
        )
    )

    assert config.resolve(checkpoint) == S3Source(
        bucket="models",
        key_prefix="shared/models/MiniMax-H3/transformer",
        region="oss-region",
        endpoint_url="https://oss.example.com",
        addressing_style="virtual",
    )


def test_storage_router_uses_longest_explicit_mount(tmp_path) -> None:
    broad = tmp_path / "mount"
    narrow = broad / "models"
    source = narrow / "wan" / "model.safetensors"
    source.parent.mkdir(parents=True)
    source.touch()
    config = build_storage_config(
        {
            "s3_mounts": [
                {
                    "mount_path": str(broad),
                    "bucket": "broad",
                    "region": "region-a",
                },
                {
                    "mount_path": str(narrow),
                    "bucket": "narrow",
                    "region": "region-b",
                },
            ]
        }
    )
    token = activate_storage_config(config)
    try:
        assert resolve_s3_source(str(source)) == S3Source(
            bucket="narrow",
            key_prefix="wan/model.safetensors",
            region="region-b",
        )
        assert resolve_s3_source(str(tmp_path / "mount-old" / "file")) is None
    finally:
        restore_storage_config(token)


def test_storage_router_resolves_read_and_write_mounts_independently(tmp_path) -> None:
    read_mount = tmp_path / "read"
    write_mount = tmp_path / "write"
    config = StorageConfig(
        s3_mounts=(
            S3MountConfig(str(read_mount), "input-bucket", "region-a"),
            S3MountConfig(str(write_mount), "output-bucket", "region-b"),
        )
    )

    assert config.resolve(read_mount / "run" / "00001.dcp") == S3Source(
        bucket="input-bucket",
        key_prefix="run/00001.dcp",
        region="region-a",
    )
    assert config.resolve(write_mount / "run" / "00002.dcp") == S3Source(
        bucket="output-bucket",
        key_prefix="run/00002.dcp",
        region="region-b",
    )


def test_storage_router_rejects_duplicate_and_relative_mounts(tmp_path) -> None:
    mount = str(tmp_path / "mount")
    with pytest.raises(ValueError, match="duplicate mount_path"):
        StorageConfig(
            s3_mounts=(
                S3MountConfig(mount, "one"),
                S3MountConfig(mount, "two"),
            )
        )
    with pytest.raises(ValueError, match="must be absolute"):
        S3MountConfig("relative", "bucket")
    with pytest.raises(ValueError, match="addressing_style"):
        S3MountConfig(mount, "bucket", addressing_style="invalid")


def test_download_tree_uses_fresh_batched_crt_transfers(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    for index in range(10):
        (source / f"shard-{index}.distcp").write_bytes(bytes([index]) * (index + 1))

    client_count = 0
    configs = []

    class FakeClient:
        def download_file(self, _bucket, key, target, *, Config):
            configs.append(Config)
            relative = key.removeprefix("prefix/")
            shutil.copyfile(source / relative, target)

    def client(_service, *, region_name):
        nonlocal client_count
        assert region_name == "region"
        client_count += 1
        return FakeClient()

    _install_fake_boto3(monkeypatch, client)

    download_tree_crt(
        str(source),
        str(destination),
        s3=S3Source(bucket="bucket", key_prefix="prefix", region="region"),
        object_workers=4,
        transfer_concurrency=32,
    )

    assert client_count == 10
    assert len(configs) == 10
    assert all(config.kwargs["max_concurrency"] == 32 for config in configs)
    for index in range(10):
        assert (destination / f"shard-{index}.distcp").read_bytes() == (
            source / f"shard-{index}.distcp"
        ).read_bytes()


def test_custom_s3_endpoint_uses_classic_multipart_transfer(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "shard.safetensors").write_bytes(b"weights")
    configs = []

    class FakeClient:
        def download_file(self, _bucket, _key, target, *, Config):
            configs.append(Config)
            shutil.copyfile(source / "shard.safetensors", target)

    def client(_service, *, region_name, endpoint_url, config):
        assert region_name == "oss-region"
        assert endpoint_url == "https://oss.example.com"
        assert config.s3["addressing_style"] == "virtual"
        return FakeClient()

    _install_fake_boto3(monkeypatch, client)
    download_tree_crt(
        str(source),
        str(destination),
        s3=S3Source(
            bucket="bucket",
            key_prefix="prefix",
            region="oss-region",
            endpoint_url="https://oss.example.com",
            addressing_style="virtual",
        ),
        object_workers=1,
        transfer_concurrency=16,
    )

    assert configs[0].kwargs["preferred_transfer_client"] == "classic"
    assert configs[0].kwargs["max_concurrency"] == 16


def test_upload_tree_returns_durable_file_metadata(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights").mkdir()
    payload = b"optimizer-payload"
    (source / "weights" / "__0_payload.distcp").write_bytes(payload)
    uploaded = {}
    configs = []

    class FakeClient:
        def upload_file(self, path, bucket, key, *, Config):
            assert bucket == "bucket"
            configs.append(Config)
            uploaded[key] = open(path, "rb").read()

        def head_object(self, *, Bucket, Key):
            assert Bucket == "bucket"
            return {"ContentLength": len(uploaded[Key])}

    def client(_service, *, region_name):
        assert region_name == "region"
        return FakeClient()

    _install_fake_boto3(monkeypatch, client)

    files = upload_tree_crt(
        str(source),
        s3=S3Source(bucket="bucket", key_prefix="run/00001.dcp", region="region"),
        object_workers=2,
        transfer_concurrency=8,
    )

    assert uploaded == {
        "run/00001.dcp/weights/__0_payload.distcp": payload,
    }
    assert files == [
        {
            "path": "weights/__0_payload.distcp",
            "size": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
    ]
    assert configs[0].kwargs["preferred_transfer_client"] == "crt"
