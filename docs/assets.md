# External asset contract

All project source is included in SPT. Standard Python/CUDA libraries are installed
as dependencies; data, weights and generated artifacts are provided through paths.

## Training/validation manifests

JSONL rows contain `sample_id`, `rgb_path` and `start_frame`. The baseline reads
a fixed 160-frame window, stride 1, at 256x256. Every window must contain enough
frames. Video paths may point to shared read-only data. Validation rows also supply
demo inputs. Preserve manifest order/content for reproduction. Preparation stores
their paths in the frozen effective configuration.

## FVD bank

Use 2048 distinct IDs (`fvd_0000` through `fvd_2047`). The JSONL lists those IDs.
Each case in the bank has:

- `ID.npy`: uint8 RGB `[160,256,256,3]`, including the observed prefix.
- `ID.json`: pixel SHA256 and original detector identity/provenance.
- `ID.feature.npy`: finite float32 `[400]` features for frames 16 through 143.

Reuse the reference bank for comparison. The first 16 frames condition generation;
frames 16:144 are scored. Preparation validates real features and detector
metadata; inference verifies input pixels on load. A changed bank changes the
experiment.

## I3D weights

Pass `i3d_torchscript.pt` to `jobs/prepare_fvd.py --weights`. Required SHA256:
`bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2`.
No code is imported from its directory. The bundled detector is byte-identical
to the reference, SHA256
`d0229a9055084132c93628edd98157fbbd5a8818f11834c27bd1a80b7e220f33`.

## Checkpoints

Use complete Native DCP (`NNNNN.dcp`) with committed `manifest.json`, weights,
optimizer and every rank's trainer state. Shared original checkpoints are read-only
inputs. New outputs/caches/frozen snapshots go below `SPT_DATA_ROOT`.
