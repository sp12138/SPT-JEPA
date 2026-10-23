# VForge test layout

- `checkpoint/`: PT/DCP persistence, conversion, and resume behavior.
- `checkpointing/`: activation checkpoint wrappers and preserved-kernel offload.
- `data/`: cache, loader, and video I/O contracts.
- `distributed/`: FSDP2, sequence parallelism, and precision policies.
- `evaluation/`: evaluation runtime behavior.
- `models/`: reusable model loading plus model-specific contracts.
- `ops/attention/`: attention dispatch, backend, debug, and offload behavior.
- `tasks/`: CLI and Hydra task entrypoints.
- `training/`: Trainer lifecycle, diagnostics, reporting, and reproducibility.
- `utils/`: shared utility behavior.
