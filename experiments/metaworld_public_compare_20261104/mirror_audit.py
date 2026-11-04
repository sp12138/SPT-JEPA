"""Search independent public Hub releases for the official MetaWorld bytes.

Only public repository metadata is read. No gated file, viewer service, proxy,
credential, or alternative endpoint for a protected file is used. Metadata
matches are candidates, not a claim that bytes have actually been downloaded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

OFFICIAL = "facebook/jepa-wms"
REVISION = "6116f042ae7ae4c8e3f1fd2f194f432615664182"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "SPT-JEPA-public-provenance-audit"})
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.load(response)


def metadata(repository, revision=None):
    url = "https://huggingface.co/api/datasets/" + repository
    if revision:
        url += "/revision/" + urllib.parse.quote(revision, safe="")
    return fetch(url + "?blobs=true")


def fingerprint(item):
    lfs = item.get("lfs") or {}
    digest = lfs.get("sha256") or lfs.get("oid")
    if digest and digest.startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    return digest if digest and len(digest) == 64 else None


def main(catalogs, output, workers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    official = metadata(OFFICIAL, REVISION)
    (output / "official-metadata.json").write_text(json.dumps(official, indent=2) + "\n")
    files = [f for f in official["siblings"] if f["rfilename"].startswith("metaworld/data/")
             and f["rfilename"].endswith(".parquet")]
    assert len(files) == 126, len(files)
    assert all(fingerprint(f) for f in files), "Official fingerprints unavailable; cannot certify a mirror"
    expected = {fingerprint(f): f["rfilename"] for f in files}
    assert len(expected) == 126
    repositories = set()
    for catalog in catalogs:
        data = json.loads(Path(catalog).read_text())
        repositories.update(x if isinstance(x, str) else x["id"] for x in data)
    repositories.discard(OFFICIAL)

    def inspect(repository):
        result = {"repository": repository}
        try:
            info = metadata(repository)
            result.update(revision=info["sha"], gated=info.get("gated"))
            if info.get("gated") is not False:
                return dict(result, status="not_ungated")
            rows = info.get("siblings", [])
            matches = [{"path": f["rfilename"], "official_path": expected[fingerprint(f)],
                        "sha256": fingerprint(f)} for f in rows if fingerprint(f) in expected]
            result.update(file_count=len(rows), exact_matches=matches,
                          matching_official_shards=len({m["sha256"] for m in matches}),
                          metaworld_like_paths=[f["rfilename"] for f in rows
                                               if "00126.parquet" in f["rfilename"]][:150],
                          status="metadata_match" if len({m["sha256"] for m in matches}) == 126
                          else "no_complete_byte_identical_mirror")
            record = json.dumps(info, sort_keys=True).encode()
            result["metadata_sha256"] = hashlib.sha256(record).hexdigest()
            if matches:
                (output / (repository.replace("/", "--") + "-metadata.json")).write_bytes(record)
        except Exception as error:
            result.update(status="metadata_error", error_type=type(error).__name__)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(inspect, name) for name in sorted(repositories)]):
            row = future.result()
            results.append(row)
            (output / (row["repository"].replace("/", "--") + ".json")).write_text(
                json.dumps(row, indent=2) + "\n")
            if row["status"] == "metadata_match" or len(results) % 20 == 0:
                print(json.dumps({"inspected": len(results), "total": len(repositories),
                                  "last": row["repository"], "status": row["status"]}), flush=True)
    report = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  official_repository=OFFICIAL, official_revision=REVISION,
                  expected_shards=len(files), inspected_repositories=len(results),
                  metadata_matches=[r for r in results if r["status"] == "metadata_match"],
                  errors=[r for r in results if r["status"] == "metadata_error"],
                  results=sorted(results, key=lambda r: r["repository"]),
                  scope="Public metadata audit only. No training or gated-file download. "
                        "Re-encoded/repacked copies cannot be excluded by a byte-fingerprint search. "
                        "Any metadata match still requires actual unauthenticated download and SHA verification.")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(args.catalog, args.output, args.workers)
