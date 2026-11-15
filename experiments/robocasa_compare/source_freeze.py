"""Create new source versions without writing through inherited symlinks."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def _clone(source, destination):
    source=source.resolve(strict=True)
    shutil.copytree(source,destination,symlinks=True)
    # Read-only dependencies may stay linked; preserve their original targets
    # even when the new version is placed at a different directory depth.
    for directory,folders,files in os.walk(destination,followlinks=False):
        for name in folders+files:
            path=Path(directory)/name
            if path.is_symlink():
                target=(source/path.relative_to(destination)).resolve(strict=True)
                path.unlink();path.symlink_to(target,target_is_directory=target.is_dir())


def _private_path(root, relative):
    current=root
    for index,part in enumerate(relative.parts):
        current=current/part
        if current.is_symlink():
            original=current.resolve(strict=True)
            if original.is_dir():
                temporary=Path(tempfile.mkdtemp(prefix='.private-',dir=current.parent))
                temporary.rmdir()
                _clone(original,temporary)
                current.unlink();temporary.rename(current)
            else:
                if index!=len(relative.parts)-1:raise NotADirectoryError(current)
                data=original.read_bytes();current.unlink();current.write_bytes(data)
        if index<len(relative.parts)-1:
            current.mkdir(exist_ok=True)
    assert current.resolve().is_relative_to(root.resolve())
    return current


def freeze_with_overrides(source, destination, overrides):
    """Clone a new version and give every overridden file private ancestors.

    Existing freezes are never edited. Dependencies not being changed remain
    read-only links. Callers publish the returned version only after validation.
    """
    source=Path(source).resolve(strict=True)
    destination=Path(destination)
    destination=destination.parent.resolve()/destination.name
    checked={}
    for name,data in overrides.items():
        relative=Path(name)
        if relative.is_absolute() or not relative.parts or '..' in relative.parts:
            raise ValueError(f'Unsafe relative source path: {name}')
        if not isinstance(data,bytes):raise TypeError('Source overrides must be bytes')
        checked[relative]=data
    if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
    if destination.is_relative_to(source):raise ValueError('Destination is inside source')
    temporary=Path(tempfile.mkdtemp(prefix='.freeze-',dir=destination.parent))
    temporary.rmdir();manifest={}
    try:
        _clone(source,temporary)
        for relative,data in checked.items():
            old=source/relative
            before=hashlib.sha256(old.read_bytes()).hexdigest() if old.exists() else None
            original=str(old.resolve())
            target=_private_path(temporary,relative)
            target.write_bytes(data)
            assert not target.is_symlink()
            for parent in target.parents:
                if parent==temporary:break
                assert not parent.is_symlink(),parent
            after=hashlib.sha256(old.read_bytes()).hexdigest() if old.exists() else None
            assert before==after,'Source changed during freeze'
            manifest[str(relative)]=dict(original_resolved=original,original_sha256=before,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),private=True)
        temporary.rename(destination)
        for relative in checked:
            assert (destination/relative).resolve().is_relative_to(destination)
        return manifest
    except BaseException:
        if temporary.exists():shutil.rmtree(temporary)
        raise
