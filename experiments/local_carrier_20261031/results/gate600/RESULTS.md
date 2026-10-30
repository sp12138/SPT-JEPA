# Local/global1024: complete600-update review

**Local spatial compression has a TwoRoom planning signal relative to dense1024, but no cross-task planning gain and no evidence yet to replace64-D Carrier.** All eight models learn factual latent dynamics; these results do not demonstrate persistent optimization failure or saturation. They also do not justify calling the larger representation a winning algorithm.

Frozen source `32fad0cc04ca46cb9e6f5516a9aa0525b65b4cea`; all40 planned endpoints exited0. Both seeds and tasks are retained. Native full-state health confirms finite model/optimizer/scheduler, expected600×128 cursor, saved RNG and portable/full-checkpoint equality. Separate CPU comparisons confirm all210 frozen tensors per run are unchanged at600, every fresh initial state equals its real preflight initial state, and each local/dense pair shares547 initial model tensors. All active/frozen gradient checks pass. Actual peak memory is107.33GiB local and107.52GiB dense.

| Task | Arm | Strict50 success, seeds1/2 | Clean rollout / initial-copy MSE | One-step / shifted-observed-copy MSE |
|---|---|---|---|---|
| TwoRoom | local1024 | 16 /24% | .806 /.828 | .841 /.837 |
| TwoRoom | dense1024 | 8 /6% | .848 /.853 | .824 /.833 |
| Cube | local1024 | 0 /4% | .774 /.765 | .831 /.828 |
| Cube | dense1024 | 4 /10% | .824 /.814 | .847 /.845 |
| TwoRoom | old64 identity Carrier | 44 /30% | .581 /.585 | .817 /.844 |
| Cube | old64 identity Carrier | 4 /6% | .528 /.518 | .735 /.735 |

The direct matched comparison is local1024 versus dense1024. Their success difference is +8/+18pp on TwoRoom, mean+13pp with shared-case bootstrap95% interval[3,23]; Cube is−4/−6pp, mean−5pp[−11,0]. Intervals condition on these two trained seeds and resample the same50 cases jointly; they do not estimate training-seed uncertainty or adjust for exploratory selection.

**Actual learning and its limits.** All clean rollouts beat initial copying, and all one-step predictions beat copying the latest observed state. For local, replacing the initial state with another episode increases rollout MSE29.6/27.8% on TwoRoom and62.6/60.2% on Cube; other-episode actions increase it4.14/4.47% and15.63/15.09%. Dense uses both inputs too, more weakly. These are real input-dependent dynamics, not solely static decoding.

That does not mean the DiT has learned to use the world state. At t=.95, local's wrong-state/correct-forecast denoising error ratios are.99875/1.00003 on TwoRoom and1.00896/1.00454 on Cube; wrong-action ratios are.99993/1.00075 and1.00549/1.00468. Dense is still less sensitive. The large flow-loss decrease can therefore mainly reflect image/anchor denoising at this budget. Clean latent prediction and renderer conditioning must remain separate claims.

From300 to600, rate still falls about9–14% across all eight runs, while flow MSE falls about90–96%. This is inside the original1249-update warmup. The factual-prediction improvements and ongoing loss changes rule out a justified claim of complete nonlearning or demonstrated saturation; one600 endpoint cannot establish an eventual ceiling. However, old64-D Carrier already learns normalized causal dynamics substantially faster at600, and local1024's Cube representation/probe gains have not translated into task success. A bounded later maturity check could answer that question, but no continuation was launched or authorized by this collection.

**Information is not summarized by one probe number.** Fixed-state validation RMS is.95–1.00 across all arms after the isotropic repair. The predefined1024-D clean ridge probe is unstable for TwoRoom local (agent R² negative, while adding the predefined sigma=.1 noise gives agent R².928/.900 and.936/.935). Its own coordinate standardization and small-sample regression can reintroduce severe conditioning/generalization sensitivity; negative clean scores cannot establish information absence. Raw probe arrays remain preserved. Cube block height R² is negative under both arms' linear probes, and this audit contains no pose-success result. No additional probe fitting or tuning was performed.

**Fairness and cost.** All40 artifacts match frozen source/config/checkpoint identities, original strict50 cases, CEM samples/horizon/iterations, observation normalization and diagnostic windows. Local/dense have identical learned parameter counts:50.248M for TwoRoom and50.310M for Cube, total55.913M/55.975M. Dense additionally stores a192MiB fixed projection, versus local12KiB. Both larger models differ from old64-D Carrier (total51.105M/51.110M) in state/rate dimension, learned capacity, fixed calibration, computation and effective metric. Local also applies spatial averaging and a different projection subspace; it is not just an invertible coordinate rotation of dense.

Recorded end-to-end planning wall seconds for50 cases, seeds1/2: TwoRoom local51.35/49.89, dense52.83/53.51, old64 Carrier44.19/46.26; Cube local95.67/94.36, dense93.87/92.80, old64 Carrier88.67/103.82. These existing records include setup and variable executed horizons and are not a controlled per-action latency assay. Equal CEM budgets do not mean equal inference FLOPs. No new timing assay was run.

Historical TF600 scalars TwoRoom6/8%, Cube14/10% are retained only as previously reported context: their original primary per-case metric artifacts were not located, so they are excluded from case-audited comparisons and confidence intervals.

Full evidence: `/efs/prescottsun/SPT-JEPA/reports/local-isotropic600-20261031/results.json`; small local `summary.json` retains all seeds, curves, probe arrays, normalized dynamics and timing. The collector reads existing artifacts and tensor equality only; it performs no forward, GPU work, fitting or evaluation rerun. Existing source/configuration/holds remain unchanged.
