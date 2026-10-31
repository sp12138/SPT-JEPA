# Cube native reset diagnostic

The fixed three-case audit completed once (exit0;33.68s process wall). Dataset-restored qpos/qvel and initial/goal pixel hashes agree across repeats. Hidden ctrl, qacc_warmstart and controller fields vary before the probe, but after the identical bounded one-step command every recorded physical/controller field is bit-exact across all9paired comparisons. This does not support a one-step dynamics divergence from the observed reset residue.

| Primary case | Episode | Restored qpos/qvel | Post-step recorded state | Distinct next-pixel hashes /3 |
|---|---:|---|---|---:|
| 3 | 937 | exact | exact | 3 |
| 28 | 6515 | exact | exact | 3 |
| 35 | 7381 | exact | exact | 3 |

Next-frame pixel hashes differ for every repeat of every case despite the recorded post-step state agreement. This warrants attention to observation/render reproducibility, but does not establish the cause of P0 flips. Only hashes were retained, so image difference magnitudes cannot be recovered. The actual native setup included18resets and36random settling steps, followed by9fixed probes; these internal steps were counted without modifying their behavior. No model, training, fullP0, reset fix or follow-up experiment was run. GPUdev2g5 was verified released.

Full arrays remain on EFS. Helper commit: `d75eb248b353c9baa05f251214876e15b4e3de71`; full-results SHA256: `52861d4c4ad558881604bc023c65321ab06bf90c35a992539a3c90f0386a0fdc`.
