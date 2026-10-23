"""Hydra entrypoint for offline DCP → PT conversion."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from vforge.tasks.common import build_and_snapshot
from vforge.training.checkpoint import (
    convert_dcp_weights_to_pt,
    estimate_dcp_conversion,
)

logger = logging.getLogger(__name__)


def run_convert_checkpoint(cfg: DictConfig) -> None:
    runtime = build_and_snapshot(cfg)
    convert = runtime.config.convert_checkpoint
    if not convert.input_path:
        raise ValueError("convert_checkpoint.input_path is required")
    if not convert.output_path and not convert.dry_run:
        raise ValueError("convert_checkpoint.output_path is required")
    logger.info(
        "converting checkpoint input=%s output=%s hash=%s",
        convert.input_path,
        convert.output_path,
        runtime.resolved_config_hash,
    )
    if convert.dry_run:
        logger.info(
            "conversion dry-run estimate=%s",
            estimate_dcp_conversion(
                convert.input_path,
                component_path=convert.component_path,
                weights=convert.weights,
            ),
        )
        return
    output = convert_dcp_weights_to_pt(
        convert.input_path,
        convert.output_path,
        component_path=convert.component_path,
        weights=convert.weights,
    )
    logger.info("convert checkpoint wrote %s", output)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="convert_checkpoint",
)
def main(cfg: DictConfig) -> None:
    run_convert_checkpoint(cfg)


if __name__ == "__main__":
    main()
