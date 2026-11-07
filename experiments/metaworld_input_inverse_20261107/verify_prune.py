"""Exercise the real cleanup function without importing the GPU trainer."""
import ast
import argparse
import errno
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch


def main(source=None):
    source = Path(source) if source else Path(__file__).parents[1]/'lingjing_fullspec_20261102/safe_train_stage.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prune_export')
    scope = dict(Path=Path)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
    prune = scope['prune_export']
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)/'model-step0002000'
        directory.mkdir(); (directory/'weights.pt').write_bytes(b'portable')
        prune(directory); assert not directory.exists()
        prune(directory)  # Recreates the observed stale-list/root-ENOENT case.
        directory.mkdir()
        for error in [FileNotFoundError(errno.ENOENT, 'Missing child', str(directory/'child')),
                      PermissionError(errno.EACCES, 'Denied'), OSError(errno.EIO, 'Storage failure')]:
            with patch.object(shutil, 'rmtree', side_effect=error):
                try:
                    prune(directory)
                except type(error):
                    pass
                else:
                    raise AssertionError('Cleanup suppressed a non-benign failure')
    print(json.dumps(dict(passed=True, checks=['existing_export_removed', 'missing_export_idempotent',
        'missing_child_not_hidden', 'permission_error_not_hidden', 'storage_error_not_hidden'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path)
    main(p.parse_args().source)
