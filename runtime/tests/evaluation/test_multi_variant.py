from __future__ import annotations

from pathlib import Path

from vforge.evaluation.multi_variant import MultiVariantOutputWriter
from vforge.evaluation.runner import EvalOutputWriter


def _file_writer(name: str) -> EvalOutputWriter:
    def write(result, output_dir):
        path = Path(output_dir) / f"{result['sample_id']}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}:{result['value']}", encoding="utf-8")
        return str(path)

    return EvalOutputWriter(
        artifact_name=lambda sample_id: f"{sample_id}.txt",
        write_output=write,
    )


def test_multi_variant_writer_preserves_variant_provenance(tmp_path) -> None:
    completion = EvalOutputWriter(
        artifact_name=lambda sample_id: f".complete/{sample_id}",
        write_output=lambda _result, output_dir: _write_completion(output_dir, "sample"),
    )
    writer = MultiVariantOutputWriter(
        variant_names=("a", "b"),
        variant_writer=_file_writer,
        completion_writer=completion,
    )
    result = {
        "variants": {
            "a": {"value": 1, "metadata": {}},
            "b": {"value": 2, "metadata": {}},
        },
        "metadata": {"shared": True},
    }
    provenance = {"checkpoint": {"id": "test"}}

    writer.write(
        result,
        str(tmp_path),
        sample_id="sample",
        seed=17,
        provenance=provenance,
    )

    assert (tmp_path / "a" / "sample.txt").read_text() == "a:1"
    assert (tmp_path / "b" / "sample.txt").read_text() == "b:2"
    assert writer.is_complete(
        str(tmp_path),
        sample_id="sample",
        seed=17,
        provenance=provenance,
    )


def _write_completion(output_dir: str, sample_id: str) -> str:
    path = Path(output_dir) / ".complete" / sample_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("complete\n", encoding="utf-8")
    return str(path)
