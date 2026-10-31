# NGC attachment-radius failure: two-seed numerical diagnosis

The two failed TwoRoom TF64 graph preparations do not contain duplicate float32 nodes. On the same saved nodes, the matrix-multiplication Euclidean-distance formula creates thousands of zero 15-neighbor radii; direct differences give strictly positive radii. This supports a numerical-cancellation explanation for the assertion, rather than exact representation collapse. It does not establish useful task information or planning quality.

| Diagnostic, 5,000 nodes per seed | Seed 1 | Seed 2 |
|---|---:|---:|
| Distinct float32 rows | 5,000 | 5,000 |
| Maximum exact multiplicity | 1 | 1 |
| Zero 15-NN radii, FP32 mm | 3,424 | 3,251 |
| Zero 15-NN radii, FP32 direct / FP64 direct | 0 / 0 | 0 / 0 |
| Minimum radius, FP64 direct | 2.7208e−5 | 3.4216e−5 |
| Median radius, FP64 direct | 1.1104e−4 | 1.2355e−4 |
| Maximum pair-distance error, FP32 mm vs direct FP64 | 4.7845e−4 | 4.6403e−4 |
| Maximum pair-distance error, FP32 direct vs direct FP64 | 2.1922e−9 | 1.7179e−9 |
| Positive-distance ordered pairs rounded to zero by mm | 966,992 | 569,302 |
| Mean squared norm | 0.21636 | 0.23396 |
| Total centered variance | 2.5328e−6 | 2.0635e−6 |

The exhaustive CPU comparison covers 25 million ordered pairs per seed. All three kernels receive exactly the same float32 node values; FP64 direct only promotes these values. CPU FP32 mm uses the official formula, but is **not** a bitwise replay of the original CUDA kernel. The original CUDA assertion and this CPU mechanism agree qualitatively; the CPU zero counts must not be presented as CUDA counts.

The mathematical distance is unchanged: `sqrt(sum((x−y)^2)) = sqrt(||x||² + ||y||² − 2 x·y)`. Here the code cloud has a common offset much larger than its local separations. Subtracting the large FP32 norm terms loses the small squared distance. Direct FP32 agrees closely with direct FP64 on all saved pairs. Centered variance is descriptive; no centered metric, floor, rescaling, or parameter search was evaluated.

## Provenance and bounded execution

- Original graph root: `/efs/prescottsun/SPT-JEPA/runs/public_ngc_nodes_20261031`; its 10 successful and 2 failed jobs remain unchanged. The public offline graph already uses FP64 direct distances; the failure is at the FP32 attachment-radius check.
- Diagnostic helper commit: `6a34b994f7cc1099591ea4eb5abc92d949c0ea08`; frozen at `runs/public_ngc_radius_source_20261031/source-v1`.
- Outputs: `runs/public_ngc_radius_diagnostic_20261031`. Both captures exit 0, exactly 10,000 RGB / 314 encoder calls total, zero predictor/action-encoder/renderer calls, no training or new cases. Encoding took 118.8 / 129.5 seconds; peak allocated memory was 349,762,048 bytes each.
- Exact same training-only 5,000-row manifest, checkpoints and actual model source as the failed jobs. Full state hashes also match across passes; parameters, persistent/nonpersistent buffers and RNG remain unchanged within each capture. See `cross-pass-provenance.json`.
- After both results completed, dev2 GPU 7 and dev13 GPU 3 were verified idle. Only owned dispatchers/monitors 184460/184461 and 1120798/1120799 were closed; closure evidence is retained. Old graph, diagnostic and all 48 evaluation jobs remain held.

No production repair or planning was performed. A future uniform direct-distance attachment implementation would be a numerically stabilized public-method variant, requiring separate frozen source and contracts, not a new algorithm. Its cost and output behavior still need review; it must cover both candidate and goal attachment, retain the original graph/radii definitions and all fixed hyperparameters, and apply to every model without result-dependent selection. The present evidence alone neither validates NGC planning nor demonstrates that TF encodings preserve control-relevant information.
