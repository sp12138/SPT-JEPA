# Uniform direct-distance NGC: all twelve numerical gates pass

All 12 derived graph checks completed with exit 0 and the frozen CPU integrity review passed. The new work used **zero RGB reads, encoder calls or other model calls**. Ten original graph NPZ files were copied byte for byte; two TwoRoom TF graphs were built from the already saved nodes with the unchanged public FP64 graph algorithm. This is a uniform numerical adaptation of a public controller, not a new learning method or a task-performance result.

| Task / model | Minimum positive 15-NN radius, seed 1 / 2 | Components before bridges, seed 1 / 2 |
|---|---:|---:|
| TwoRoom local1024 | 0.745029 / 0.732298 | 3 / 2 |
| TwoRoom native raw | 0.633588 / 0.602797 | 1 / 1 |
| TwoRoom TF64 | 2.72080e−5 / 3.42156e−5 | 33 / 31 |
| Cube local1024 | 12.684570 / 12.333879 | 22 / 28 |
| Cube native raw | 5.206442 / 4.692369 | 1 / 1 |
| Cube TF64 | 0.564187 / 0.626438 | 1 / 2 |

Every saved node passed finite, strictly positive radius and exact self-goal-zero checks. Every graph retained exactly components−1 public-rule bridges. The table reports each representation's own units, not a cross-family distance-quality comparison. Disconnection/bridging alone does not establish planning quality or physical reachability.

All 25 million ordered node pairs per model were compared on CUDA: FP32 direct differences versus FP64 direct differences on the same promoted float32 nodes. Across all 12 models, maximum relative error was 1.851e−7, comfortably below the fixed dimension-derived arithmetic bound. Absolute errors depend on representation scale; all values are retained in `graph-gate.json`. No floor or fitted correction was applied. Checks took 7.1–26.4 seconds per job after provenance setup, with 121–258 MB peak allocated device memory; these are graph-gate costs, not planning latency.

The two TF controls now also reproduce the **actual CUDA-mm** failure: seed 1 / 2 had **3,410 / 3,230** zero 15-NN radii and the original public `attach` asserted in all 157 chunks for each seed. This is separate from the previous CPU-mm counts 3,424 / 3,251. FP32 direct distance gave no zero radii, while all 5,000 nodes per seed remained unique. Thus the concrete numerical failure reproduces on the original device; no exact-node-collapse conclusion is justified.

`StableNGCCost` only overrides attachment distance computation. It retains FP32, stable sorting, k=15, anchors=4, inherited route/direct costs and goal caching. Public vendor files are unchanged. The same implementation applies to every model. Passing node queries does not guarantee every future predicted query; live positive-radius and finite assertions remain active, with no fallback. Both L2 and NGC arms calculate both scores for shared-bank diagnostics, so reported wall time will be instrumented rather than bare-L2 latency.

## Preserved failure and recovery

The first source `a2076777ae85342b62af1449543e849da4da350d` completed its mathematical loops but all 12 jobs exited 1 when a NumPy float32 report field could not serialize to JSON. Source, failure exits and saved NPZ arrays remain untouched in `runs/public_ngc_stable_nodes_20261031`. They were never relabeled successful.

Source `05ebb20bc564fdaa7a79533f5c2d62c1b5983412` changes only that field's Python `float` conversion. A CPU contract evaluates the actual report expression with all 12 real saved fixtures, serializes every field, reproduces rejection of the original NumPy scalar, and rejects NaN/±Inf. All 12 pass. Unsaved timing/error counters in this serialization-only contract use neutral values of the actual Python types; none are claimed as recovered measurements. The separate 12 original numerical/controller contracts also passed.

Successful derived root: `/efs/prescottsun/SPT-JEPA/runs/public_ngc_stable_nodes_v2_20261031`. Frozen helper: `runs/public_ngc_stable_source_20261031/source-v2`. The all-graph gate SHA256 is `2425c61ed060ed1a246dd40abc7416e27ee344ddbf8788a1538cb62c98496ae3`. All checkpoints, original model sources, cohorts, graph/node hashes and original full-state hashes are bound in the reports. The node-only workers/monitors were verified idle, closed and held.

After root review, the separately frozen 48 paired evaluations were activated at 09:23:46 UTC in `runs/public_ngc_stable_evaluation_v2_20261031`, with 16 reserved GPUs. All original failed/held roots remain intact. No planning outcomes have been analyzed here. The protocol retains contemporary L2 replays, all seeds, both case sets, local1024 on both tasks, raw native, TF, original command CEM and strict evaluation; no action covariance/projection or historical-best selection is included.
