"""Fetch pinned public RoboCasa inputs; never uses credentials or private mirrors."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request
import zipfile


SOURCES = {
    "robo_jepa": ("facebookresearch/robo_jepa", "f93106f5f4a451fd526d51778aa8ab7c852eb824"),
    "robocasa": ("robocasa/robocasa", "456174f62b89b8fca99eaaf33949c29fec9cfc2a"),
    "robosuite": ("ARISE-Initiative/robosuite", "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"),
    "vjepa2": ("facebookresearch/vjepa2", "204698b45b3712590f06245fbfba32d3be539812"),
}
TASKS = ["OpenDrawer", "CloseDrawer", "PickPlaceCounterToCabinet",
         "PickPlaceCabinetToCounter", "PickPlaceCounterToSink", "PickPlaceSinkToCounter"]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(argv, cwd=None):
    subprocess.run(argv, cwd=cwd, check=True, timeout=900)


def sources(root):
    records = {}
    for name, (repo, revision) in SOURCES.items():
        path = root / "sources" / name
        if not (path / ".git").exists():
            path.mkdir(parents=True, exist_ok=False)
            command(["git", "init", "-q", str(path)])
            command(["git", "remote", "add", "origin", "https://github.com/" + repo + ".git"], path)
            if name == "robo_jepa":
                command(["git", "sparse-checkout", "set", "--no-cone", "/app/", "/src/",
                         "/scripts/", "/configs/", "/environment.yml", "/README.md", "/LICENSE"], path)
            command(["git", "fetch", "--depth", "1", "origin", revision], path)
            command(["git", "checkout", "--detach", "FETCH_HEAD"], path)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        assert actual == revision, (name, actual, revision)
        records[name] = dict(path=str(path), revision=actual, repository=repo)
    save(root / "sources.json", records)
    return records


def download(url, path, max_bytes=80_000_000_000):
    path = Path(path)
    proof = path.with_suffix(path.suffix + ".download.json")
    if path.exists() and proof.exists():
        record = json.loads(proof.read_text())
        assert path.stat().st_size == record["bytes"] and sha(path) == record["sha256"]
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        try:
            offset = part.stat().st_size if part.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=90) as response:
                assert response.status in (200, 206), response.status
                if offset and response.status != 206:
                    offset = 0
                expected = int(response.headers.get("Content-Length", 0)) + offset
                assert expected <= max_bytes, (path.name, expected)
                kind = response.headers.get("Content-Type", "")
                assert "text/html" not in kind, (path.name, kind)
                count = offset
                with part.open("ab" if offset else "wb") as output:
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        count += len(block)
                        assert count <= max_bytes
                        output.write(block)
                assert not expected or count == expected, (count, expected)
                record = dict(url=url, bytes=count, sha256=sha(part), content_type=kind,
                              etag=response.headers.get("ETag"), attempt=attempt + 1)
            part.replace(path)
            save(proof, record)
            return record
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def unpack(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    if str(archive).endswith(".zip"):
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                assert (base / member.filename).resolve().is_relative_to(base), member.filename
                assert (member.external_attr >> 16) & 0o170000 != 0o120000
            stream.extractall(destination)
    else:
        with tarfile.open(archive) as stream:
            stream.extractall(destination, filter="data")


def fetch_assets(root):
    package = root / "sources/robocasa/robocasa"
    links = json.loads((package / "models/assets/box_links/box_links_assets.json").read_text())
    groups = {"textures": "textures", "generative_textures": "generative_textures",
              "fixtures_lightwheel": "fixtures", "objaverse": "objects/objaverse",
              "aigen_objs": "objects/aigen_objs", "objects_lightwheel": "objects/lightwheel"}

    def one(key, relative):
        archive = root / "downloads/assets" / (key + ".zip")
        url = links[key].replace("/s/", "/shared/static/") + ".zip"
        record = download(url, archive)
        marker = archive.with_suffix(".extracted.json")
        if not marker.exists():
            # The official ZIPs include their top-level directory.
            unpack(archive, (package / "models/assets" / relative).parent)
            save(marker, dict(sha256=record["sha256"], destination=str(package / "models/assets" / relative)))
        print(json.dumps(dict(kind="asset", group=key, bytes=record["bytes"], passed=True)), flush=True)
        return key, record

    records = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(one, k, v) for k, v in groups.items()]):
            key, record = future.result()
            records[key] = record
    save(root / "assets.json", records)


def fix_asset_layout(root):
    """Move the initial bootstrap's nested ZIP roots into their official location."""
    assert (root / "assets.json").exists(), "Wait for the original extraction to finish"
    base = root / "sources/robocasa/robocasa/models/assets"
    relative = ["textures", "generative_textures", "fixtures", "objects/objaverse",
                "objects/aigen_objs", "objects/lightwheel"]
    moves = []

    def merge(source, destination):
        for child in list(source.iterdir()):
            target = destination / child.name
            if child.is_dir() and target.is_dir():
                merge(child, target)
            else:
                child.replace(target)
        source.rmdir()

    for name in relative:
        outer = base / name
        inner = outer / outer.name
        if inner.is_dir():
            merge(inner, outer)
            moves.append(name)
        assert outer.is_dir() and any(outer.iterdir()), name
    save(root / "asset-layout.json", dict(passed=True, corrected_nested_roots=moves,
          source_archives_unchanged=True, no_training_or_simulator_outcome=True))


def fetch_data(root):
    catalog_path = root / "sources/robocasa/robocasa/models/assets/box_links/box_links_ds.json"
    catalog = json.loads(catalog_path.read_text())

    def one(task):
        keys = [k for k in catalog if k.startswith("pretrain/atomic/" + task + "/")
                and "/mg/" not in k and k.endswith("/lerobot.tar")]
        assert len(keys) == 1, (task, keys)
        key = keys[0]
        archive = root / "downloads/data" / (task + ".tar")
        record = download(catalog[key].replace("/s/", "/shared/static/") + ".tar", archive)
        destination = root / "data" / task
        marker = destination / "EXTRACTED.json"
        if not marker.exists():
            unpack(archive, destination)
            save(marker, dict(catalog_key=key, sha256=record["sha256"]))
        print(json.dumps(dict(kind="training_data", task=task, bytes=record["bytes"], passed=True)), flush=True)
        return task, dict(catalog_key=key, archive=record, destination=str(destination))

    records = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(one, task) for task in TASKS]):
            key, record = future.result()
            records[key] = record
    save(root / "data-downloads.json", records)


def fetch_reference(root):
    """Released models are external comparators, never candidate teachers/inits."""
    reference = root / "reference"
    reference.mkdir(exist_ok=True)
    save(reference / "REFERENCE_ONLY.json", dict(external_pretraining=True,
         forbidden_candidate_initialization=True, forbidden_teacher=True,
         purpose="Frozen public RoboJEPA performance reference; unequal training data/prior"))
    downloads = [("encoder", "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt", reference / "encoder.pth.tar")]
    for size, step in [("50m", 120), ("2b", 300)]:
        model = reference / size
        model.mkdir(exist_ok=True)
        source = root / "sources/robo_jepa/configs/released" / ("wm-" + size) / "config.json"
        (model / "config.json").write_bytes(source.read_bytes())
        link = model / "encoder.pth.tar"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(reference / "encoder.pth.tar")
        downloads.append((size, f"https://dl.fbaipublicfiles.com/robojepa/wm/wm-{size}/cooldown_{size}_{step}k.pth.tar",
                          model / "world_model.pth.tar"))

    def one(item):
        name, url, path = item
        result = download(url, path, max_bytes=40_000_000_000)
        print(json.dumps(dict(reference=name, bytes=result["bytes"], passed=True)), flush=True)
        return name, result

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = dict(pool.map(one, downloads))
    save(root / "reference-downloads.json", records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["sources", "assets", "data", "fix-assets", "reference"], required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        {"sources": sources, "assets": fetch_assets, "data": fetch_data,
         "fix-assets": fix_asset_layout, "reference": fetch_reference}[args.mode](args.root)
    except BaseException as error:
        save(args.root / ("acquire-" + args.mode + "-error.json"), dict(error=repr(error)))
        raise


if __name__ == "__main__":
    main()
