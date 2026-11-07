"""Test the real runtime-record writer's repeated and concurrent attempts."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import tempfile
from unittest.mock import patch


def main(source=None):
    source = Path(source) if source else Path(__file__).parents[2]/'as_jepa_tools/train_stage.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'write_runtime_source')
    scope = dict(Path=Path, json=json)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
    write = scope['write_runtime_source']
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); original = dict(step=1001, sha256={'model':'unchanged'})
        first = write(root, 5000, 1000, original); payload = first.read_bytes()
        rng = random.getstate()
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(lambda n: write(root, 5000, 4000, dict(attempt=n)), range(24)))
        assert first.read_bytes() == payload
        assert len(set(paths)) == 24 and first not in paths
        assert {json.loads(p.read_text())['attempt'] for p in paths} == set(range(24))
        assert random.getstate() == rng
        assert all('-resume0004000-' in p.name for p in paths)
        with patch.object(Path, 'open', side_effect=PermissionError('Denied')):
            try: write(root, 5000, 4000, {})
            except PermissionError: pass
            else: raise AssertionError('Permission failure was hidden')
    print(json.dumps(dict(passed=True, checks=['original_provenance_preserved', '24_unique_concurrent_attempts',
        'resume_step_identified', 'python_rng_unchanged', 'non_collision_error_propagated'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path)
    main(p.parse_args().source)
