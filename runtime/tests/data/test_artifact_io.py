from __future__ import annotations

import hashlib
import multiprocessing
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vforge.data.artifact_io import ArtifactStore
from vforge.utils import s3_staging

URI = "s3://test-bucket/nested/sample.pt"


class FakeS3:
    def __init__(self) -> None:
        self.content = b"sample-tensor-bytes"
        self.version: str | None = None
        self.etag = '"etag-first"'
        self.head_calls = 0
        self.download_calls = 0
        self.download_kwargs = []
        self.change_on_download = False
        self.lock = threading.Lock()

    def head_object(self, *, Bucket, Key):
        assert Bucket == "test-bucket"
        assert Key == "nested/sample.pt"
        with self.lock:
            self.head_calls += 1
        result = {"ContentLength": len(self.content), "ETag": self.etag}
        if self.version is not None:
            result["VersionId"] = self.version
        return result

    def download_file(self, bucket, key, target, **kwargs):
        assert bucket == "test-bucket"
        assert key == "nested/sample.pt"
        with self.lock:
            self.download_calls += 1
            self.download_kwargs.append(kwargs)
        Path(target).write_bytes(self.content)
        if self.change_on_download:
            self.etag = '"etag-changed-during-download"'


@pytest.fixture
def remote(monkeypatch):
    client = FakeS3()
    monkeypatch.setattr(s3_staging, "_s3_client", lambda source: client)
    monkeypatch.setattr(s3_staging, "_transfer_config", lambda *args, **kwargs: None)
    return client


def test_local_identity_and_contract_ignore_cache_location(tmp_path):
    path = tmp_path / "tensor.pt"
    path.write_bytes(b"local sample")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    left, right = ArtifactStore(tmp_path / "a"), ArtifactStore(tmp_path / "b")
    assert left.resolve(path, digest.upper()) == path
    assert left.identity(path) == {
        "uri": str(path),
        "sha256": "sha256:" + digest,
        "size": len(b"local sample"),
    }
    assert left.manifest_contract([path, path]) == right.manifest_contract([path])
    assert left.manifest_contract([path])["artifact_count"] == 1


def test_frozen_local_identity_rejects_same_size_edit(tmp_path):
    path = tmp_path / "tensor.pt"
    path.write_bytes(b"first")
    store = ArtifactStore(tmp_path / "cache")
    before = store.manifest_contract([path])
    path.write_bytes(b"other")
    with pytest.raises(ValueError, match="changed during dataset session"):
        store.resolve(path)
    assert ArtifactStore().manifest_contract([path]) != before


def test_declared_digest_is_enforced_on_first_and_cached_read(tmp_path, remote):
    store = ArtifactStore(tmp_path / "cache")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        store.resolve(URI, "0" * 64)
    assert not list(store.cache_dir.rglob("artifact.pt"))
    store.resolve(URI)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        store.resolve(URI, "1" * 64)
    with pytest.raises(ValueError, match="digest must be"):
        store.resolve(URI, "md5:invalid")


def test_s3_freezes_session_but_new_store_checks_head(tmp_path, remote):
    cache = tmp_path / "cache"
    store = ArtifactStore(cache)
    local = store.resolve(URI)
    assert local.read_bytes() == remote.content
    assert remote.download_calls == 1
    assert remote.head_calls == 2
    store.identity(URI)
    store.resolve(URI)
    assert remote.head_calls == 2
    assert ArtifactStore(cache).resolve(URI) == local
    assert remote.head_calls == 3
    assert remote.download_calls == 1


def test_remote_change_selects_new_bytes_and_contract(tmp_path, remote):
    cache = tmp_path / "cache"
    original = ArtifactStore(cache)
    before = original.manifest_contract([URI])
    old_local = original.resolve(URI)
    remote.content = b"updated remote file"
    remote.etag = '"etag-new"'
    fresh = ArtifactStore(cache)
    new_local = fresh.resolve(URI)
    assert new_local != old_local
    assert new_local.read_bytes() == remote.content
    assert original.manifest_contract([URI]) == before
    assert fresh.manifest_contract([URI]) != before
    assert remote.download_calls == 2


def test_version_change_changes_contract_even_for_identical_bytes(tmp_path, remote):
    remote.version = "version-one"
    first = ArtifactStore(tmp_path / "cache").manifest_contract([URI])
    assert remote.download_kwargs[0]["ExtraArgs"] == {"VersionId": "version-one"}
    remote.version = "version-two"
    second = ArtifactStore(tmp_path / "cache").manifest_contract([URI])
    assert first != second
    assert remote.download_kwargs[1]["ExtraArgs"] == {"VersionId": "version-two"}


def test_changed_unversioned_object_is_not_published(tmp_path, remote):
    remote.version = "null"  # A suspended-versioning object is still mutable.
    remote.change_on_download = True
    store = ArtifactStore(tmp_path / "cache")
    with pytest.raises(ValueError, match="changed during download"):
        store.resolve(URI)
    assert not list(store.cache_dir.rglob("artifact.pt"))
    assert "ExtraArgs" not in remote.download_kwargs[0]


def test_corrupt_cache_is_rejected_by_session_and_repaired_by_new_store(tmp_path, remote):
    cache = tmp_path / "cache"
    store = ArtifactStore(cache)
    path = store.resolve(URI)
    path.write_bytes(b"x" * len(remote.content))
    with pytest.raises(ValueError, match="changed during dataset session"):
        store.resolve(URI)
    repaired = ArtifactStore(cache).resolve(URI)
    assert repaired == path
    assert repaired.read_bytes() == remote.content
    assert remote.download_calls == 2


@pytest.mark.parametrize("metadata", ["broken-json", "null", "{}"])
def test_partial_cache_metadata_is_rebuilt(tmp_path, remote, metadata):
    cache = tmp_path / "cache"
    path = ArtifactStore(cache).resolve(URI)
    path.with_name("identity.json").write_text(metadata)
    assert ArtifactStore(cache).resolve(URI).read_bytes() == remote.content
    assert remote.download_calls == 2


def test_simultaneous_stores_share_one_atomic_download(tmp_path, remote):
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: ArtifactStore(cache).resolve(URI), range(16)))
    assert len(set(paths)) == 1
    assert all(path.read_bytes() == remote.content for path in paths)
    assert remote.download_calls == 1
    assert not list(cache.rglob("download-*"))


def _process_resolve(cache, download_log):
    client = FakeS3()
    download = client.download_file

    def traced_download(*args, **kwargs):
        with Path(download_log).open("a") as handle:
            handle.write("download\n")
        download(*args, **kwargs)

    client.download_file = traced_download
    s3_staging._s3_client = lambda source: client
    s3_staging._transfer_config = lambda *args, **kwargs: None
    assert ArtifactStore(cache).resolve(URI).read_bytes() == client.content


def test_independent_processes_share_one_download(tmp_path):
    cache, log = tmp_path / "cache", tmp_path / "downloads.log"
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_process_resolve, args=(cache, log)) for _ in range(4)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
        assert [process.exitcode for process in processes] == [0, 0, 0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    assert log.read_text().splitlines() == ["download"]


def test_failed_download_leaves_no_published_object(tmp_path, remote, monkeypatch):
    def fail(bucket, key, target, **kwargs):
        Path(target).write_bytes(b"partial")
        raise OSError("download interrupted")

    monkeypatch.setattr(remote, "download_file", fail)
    store = ArtifactStore(tmp_path / "cache")
    with pytest.raises(OSError, match="download interrupted"):
        store.resolve(URI)
    assert not list(store.cache_dir.rglob("artifact.pt"))
    assert not list(store.cache_dir.rglob("download-*"))


def test_frozen_store_is_pickle_safe_and_worker_reads_need_no_client(tmp_path, remote, monkeypatch):
    store = ArtifactStore(tmp_path / "cache")
    contract = store.manifest_contract([URI])
    worker_store = pickle.loads(pickle.dumps(store))

    def no_network(source):
        raise AssertionError("frozen worker must not create an S3 client")

    monkeypatch.setattr(s3_staging, "_s3_client", no_network)
    assert worker_store.resolve(URI).read_bytes() == remote.content
    assert worker_store.manifest_contract([URI]) == contract


@pytest.mark.parametrize(
    "uri",
    ["https://example.com/file.pt", "s3://bucket", "s3:///key", "s3://bucket/key?versionId=x"],
)
def test_unsupported_or_ambiguous_uri_fails_before_network(tmp_path, uri):
    with pytest.raises(ValueError):
        ArtifactStore(tmp_path / "cache").resolve(uri)
