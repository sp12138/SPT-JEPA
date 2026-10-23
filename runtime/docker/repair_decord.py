"""Repair decord 0.6.0's installed wheel metadata after uv verifies its archive.

The Linux wheel is named py3-none but declares cp36-cp36m internally. It uses
ctypes, not a CPython extension. Only WHEEL and RECORD change; the dependency
keeps its registry identity so a subsequent uv sync does not reinstall it.
"""

import base64
import csv
import hashlib
import json
import sys
from importlib.metadata import distribution
from pathlib import Path


def repair(manifest: Path) -> None:
    package = distribution("decord")
    assert package.version == "0.6.0", "Recheck the workaround before upgrading decord"
    files = list(package.files or [])
    wheel = next(f for f in files if str(f).endswith(".dist-info/WHEEL"))
    record = next(f for f in files if str(f).endswith(".dist-info/RECORD"))
    path = package.locate_file(wheel)
    old = "Tag: cp36-cp36m-manylinux2010_x86_64"
    new = "Tag: py3-none-manylinux2010_x86_64"
    original = path.read_text()
    assert original.count(old) == 1 or original.count(new) == 1
    path.write_text(original.replace(old, new))
    # Upstream RECORD also has stale hashes. Regenerate from the installed
    # payload, authenticated by uv against the archive hash in uv.lock.
    with package.locate_file(record).open("w", newline="") as stream:
        writer = csv.writer(stream)
        for entry in files:
            if entry == record:
                writer.writerow([str(entry), "", ""])
            else:
                data = package.locate_file(entry).read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
                writer.writerow([str(entry), "sha256=" + digest.decode(), len(data)])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"version": package.version, "correction": f"{old} -> {new}; payload unchanged"})
        + "\n"
    )


if __name__ == "__main__":
    repair(Path(sys.argv[1]))
