"""Model-independent orchestration for evaluating named sample variants."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from vforge.evaluation.runner import EvalInference, EvalOutputWriter


class NamedVariant(Protocol):
    """Minimum variant identity required by the generic orchestrator."""

    name: str


class MultiVariantSharedOutput(Protocol):
    """Optional model-owned shared artifacts written once per sample."""

    def is_complete(self, output_dir: str, *, sample_id: str) -> bool: ...

    def write(
        self,
        result: Mapping[str, Any],
        output_dir: str,
        *,
        sample_id: str,
    ) -> None: ...


VariantT = TypeVar("VariantT", bound=NamedVariant)
PrepareVariant = Callable[[dict[str, Any], VariantT], dict[str, Any]]


@dataclass
class MultiVariantInference(Generic[VariantT]):
    """Apply model-owned sample transforms around one inference implementation."""

    base: EvalInference
    variants: Sequence[VariantT]
    prepare_variant: PrepareVariant[VariantT]

    def infer_sample(self, sample: dict[str, Any], *, seed: int) -> dict[str, Any]:
        outputs: dict[str, dict[str, Any]] = {}
        for variant in self.variants:
            name = str(variant.name)
            prepared = self.prepare_variant(sample, variant)
            result = self.base.infer_sample(prepared, seed=int(seed))
            if not isinstance(result, dict):
                raise TypeError(
                    f"variant inference must return a dictionary, got {type(result).__name__}"
                )
            result["variant"] = name
            outputs[name] = result
        return {
            "sample_id": sample.get("sample_id"),
            "seed": int(seed),
            "variants": outputs,
            "metadata": {
                "variants": [str(variant.name) for variant in self.variants],
            },
        }


class MultiVariantOutputWriter:
    """Write every named result before publishing one completion artifact."""

    def __init__(
        self,
        *,
        variant_names: Sequence[str],
        variant_writer: Callable[[str], EvalOutputWriter],
        completion_writer: EvalOutputWriter,
        shared_output: MultiVariantSharedOutput | None = None,
        provenance_variant_key: str = "evaluation_variant",
    ) -> None:
        names = tuple(str(name) for name in variant_names)
        if not names or len(set(names)) != len(names):
            raise ValueError(f"variant_names must be non-empty and unique, got {list(names)}")
        self.variant_names = names
        self._variant_writers = {name: variant_writer(name) for name in names}
        self._completion = completion_writer
        self._shared_output = shared_output
        self._provenance_variant_key = str(provenance_variant_key)

    def expected_path(self, output_dir: str, sample_id: str) -> str:
        return self._completion.expected_path(output_dir, sample_id)

    def is_complete(
        self,
        output_dir: str,
        *,
        sample_id: str,
        seed: int,
        provenance: Mapping[str, Any],
    ) -> bool:
        if not self._completion.is_complete(
            output_dir,
            sample_id=sample_id,
            seed=seed,
            provenance=provenance,
        ):
            return False
        variants_complete = all(
            self._variant_writers[name].is_complete(
                os.path.join(output_dir, name),
                sample_id=sample_id,
                seed=seed,
                provenance={**provenance, self._provenance_variant_key: name},
            )
            for name in self.variant_names
        )
        shared_complete = self._shared_output is None or self._shared_output.is_complete(
            output_dir,
            sample_id=sample_id,
        )
        return variants_complete and shared_complete

    def write(
        self,
        result: dict[str, Any],
        output_dir: str,
        *,
        sample_id: str,
        seed: int,
        provenance: Mapping[str, Any],
    ) -> str:
        variants = result.get("variants")
        if not isinstance(variants, Mapping) or not variants:
            raise ValueError("multi-variant result carries no variant outputs")
        if set(variants) != set(self.variant_names):
            raise ValueError(
                "result variants do not match the requested set: "
                f"expected={list(self.variant_names)} actual={list(variants)}"
            )
        for name, variant_result in variants.items():
            payload = dict(variant_result)
            payload["sample_id"] = sample_id
            payload["metadata"] = {
                **dict(result.get("metadata") or {}),
                **dict(payload.get("metadata") or {}),
                "variant": str(name),
            }
            self._variant_writers[str(name)].write(
                payload,
                os.path.join(output_dir, str(name)),
                sample_id=sample_id,
                seed=seed,
                provenance={**provenance, self._provenance_variant_key: str(name)},
            )
        if self._shared_output is not None:
            self._shared_output.write(
                result,
                output_dir,
                sample_id=sample_id,
            )
        completion_payload = {**result, "sample_id": sample_id}
        return self._completion.write(
            completion_payload,
            output_dir,
            sample_id=sample_id,
            seed=seed,
            provenance=provenance,
        )


__all__ = [
    "MultiVariantInference",
    "MultiVariantOutputWriter",
    "MultiVariantSharedOutput",
    "NamedVariant",
]
