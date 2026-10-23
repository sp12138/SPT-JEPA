from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import ModuleType

import pytest
from botocore.exceptions import ClientError

import vforge.training.checkpoint.remote_save as remote_save
from vforge.training.checkpoint.manifest import CheckpointManifest
from vforge.training.checkpoint.remote_save import (
    RemoteSaveContext,
    prepare_remote_save,
    upload_and_commit_remote_save,
)
from vforge.utils.s3_staging import (
    S3MountConfig,
    S3Source,
    StorageConfig,
    activate_storage_config,
    restore_storage_config,
)


def test_remote_save_cleans_local_tree_after_failure(monkeypatch, tmp_path) -> None:
    local = tmp_path / "upload.incomplete"
    local.mkdir()
    (local / "payload").write_bytes(b"data")
    context = RemoteSaveContext(
        remote_path="/mount/run/00001.dcp",
        local_incomplete=str(local),
        s3=S3Source("bucket", "run/00001.dcp", "region"),
        node_rank=0,
        local_rank=0,
    )
    monkeypatch.setattr(
        remote_save,
        "_upload_and_commit_remote_save",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    with pytest.raises(RuntimeError, match="failed"):
        upload_and_commit_remote_save(
            context,
            CheckpointManifest.create(state="weights", completed_steps=1),
            process_group=None,
        )

    assert not local.exists()


def test_remote_save_uploads_node_tree_and_commits_manifest_last(
    monkeypatch,
    tmp_path,
) -> None:
    mount = tmp_path / "s3"
    remote = mount / "checkpoints" / "00001.dcp"
    uploaded = {}
    events = []

    class FakePaginator:
        def paginate(self, **_kwargs):
            return [{"Contents": []}]

    class FakeClient:
        def head_object(self, *, Bucket, Key):
            assert Bucket == "bucket"
            if Key not in uploaded:
                raise ClientError(
                    {"Error": {"Code": "404", "Message": "missing"}},
                    "HeadObject",
                )
            return {"ContentLength": len(uploaded[Key])}

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return FakePaginator()

        def delete_objects(self, **_kwargs):
            raise AssertionError("empty prefix should not issue deletes")

        def upload_file(self, path, bucket, key, *, Config):
            assert bucket == "bucket"
            assert Config.kwargs["preferred_transfer_client"] == "crt"
            uploaded[key] = open(path, "rb").read()
            events.append(("payload", key))

        def put_object(self, *, Bucket, Key, Body, ContentType):
            assert Bucket == "bucket"
            assert ContentType in {"application/json", "text/plain"}
            uploaded[Key] = bytes(Body)
            events.append(("commit", Key))

    class FakeTransferConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    boto3 = ModuleType("boto3")
    boto3.client = lambda *_args, **_kwargs: FakeClient()
    boto3_s3 = ModuleType("boto3.s3")
    boto3_transfer = ModuleType("boto3.s3.transfer")
    boto3_transfer.TransferConfig = FakeTransferConfig
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "boto3.s3", boto3_s3)
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", boto3_transfer)
    monkeypatch.setenv("VFORGE_CKPT_LOCAL_ROOT", str(tmp_path / "local"))
    monkeypatch.setenv("VFORGE_NODE_IO_INLINE", "1")

    token = activate_storage_config(
        StorageConfig(s3_mounts=(S3MountConfig(str(mount), "bucket", "region"),))
    )
    context = prepare_remote_save(str(remote), process_group=None)
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    trainer = tmp_path / "trainer.pt"
    trainer.write_bytes(b"trainer")
    weights_dir = context.local_incomplete + "/weights"
    trainer_dir = context.local_incomplete + "/trainer"
    import os

    os.makedirs(weights_dir)
    os.makedirs(trainer_dir)
    open(weights_dir + "/__0_payload.distcp", "wb").write(weights.read_bytes())
    open(trainer_dir + "/rank-00000.pt", "wb").write(trainer.read_bytes())
    manifest = replace(
        CheckpointManifest.create(state="full", completed_steps=1),
        components={
            "weights": "weights",
            "optimizer": "optimizer",
            "trainer": "trainer",
        },
        components_present={
            "weights": True,
            "optimizer": False,
            "trainer": True,
        },
    )

    committed = upload_and_commit_remote_save(
        context,
        manifest,
        process_group=None,
    )

    manifest_key = "checkpoints/00001.dcp/manifest.json"
    assert manifest_key in uploaded
    assert events[-2:] == [
        ("commit", manifest_key),
        ("commit", "checkpoints/latest"),
    ]
    payload = json.loads(uploaded[manifest_key])
    assert (
        payload["fingerprints"]["component_integrity"]
        == (committed.fingerprints["component_integrity"])
    )
    assert committed.fingerprints["component_integrity"]["weights"]["total_bytes"] == len(
        b"weights"
    )
    restore_storage_config(token)
    assert not os.path.exists(context.local_incomplete)
