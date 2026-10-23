"""The data layer may not reach back into task assembly.

``vforge/data`` produces samples and batches; ``vforge/tasks`` reads a request,
picks a source, and derives the resume fingerprint. Keeping the arrow pointing
one way is what lets corpus-specific modules such as ``gameplay_dataset`` sit
under ``data/`` at all -- the moment one of them imports a request type or a
task builder the two layers are one layer with two directories.

An import graph erodes a comment but not an assertion, so this is a test.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "vforge" / "data"


def _absolute_imports(source: str) -> set[str]:
    """Every absolute module a file imports, lazy function-body ones included."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def test_the_data_layer_never_imports_task_assembly() -> None:
    modules = sorted(DATA_ROOT.rglob("*.py"))
    assert modules, f"no data modules found under {DATA_ROOT}"

    offenders = {
        str(path.relative_to(REPO_ROOT)): sorted(reaching_back)
        for path in modules
        if (
            reaching_back := {
                module
                for module in _absolute_imports(path.read_text(encoding="utf-8"))
                if module == "vforge.tasks" or module.startswith("vforge.tasks.")
            }
        )
    }

    assert not offenders, (
        "vforge/data produces samples and vforge/tasks assembles them; "
        f"these import the wrong way round: {offenders}"
    )
