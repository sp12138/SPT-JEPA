# Fixed action-correlation CEM control

This is a known-controller mechanism control, not a new learning algorithm. The prior Cube fixed-pool evidence is in RESULTS.md. No learned-world-model objective, weights, metric, optimal-action supervision or conditional behavior policy changes.

## Frozen training statistics

Exactly the pre-existing train128 cohorts are used. Per clip, only its first25 raw actions enter the estimate; Cube reuses the verified64,000-byte action cache and TwoRoom reads25,600 bytes plus row/episode metadata. Both are128 distinct training episodes, disjoint from validation. No RGB or task labels enter calibration. Neither dataset has zero empirical variance in any trajectory coordinate; preparation would stop on such a coordinate before shrinkage, without adding an exploration assumption.

Normalize by the existing train-card per-channel scaler. Flatten in raw-step-major, channel-minor order. Estimate Ledoit–Wolf C on these128 vectors; set R=diag(C)^(-1/2) C diag(C)^(-1/2), then use **positive-diagonal Cholesky** L, LLᵀ=R. No dataset mean is added to proposals. No eigenfloor, truncation, noise-color selection, coefficient search or post-outcome change is allowed. The old diagnostic's symmetric square-root field is not used.

| Task | Dimension | LW shrinkage | Empirical participation rank | R participation rank | R eigenvalue range | Lag1 per channel | High-frequency DCT power |
|---|---:|---:|---:|---:|---|---|---:|
| TwoRoom |50|.843797|34.42|49.46|.8650–1.3183|.010/.054|69.61%|
| Cube |125|.151934|14.04|19.57|.1321–16.9358|.802/.838/.947/.840/.906|9.91%|

The identical rule therefore yields near-independent TwoRoom exploration and a substantially correlated Cube search. The contrast follows training data, not a manual task switch. Cholesky depends on coordinate order; its statistical innovations are not identified causal actuator mechanisms. L and LQ have the same initial Gaussian law for orthogonal Q, but diagonal elite adaptation generally makes later searches different. The factorization choice is therefore frozen, not dismissed as irrelevant.

Calibration NPZ hashes:

- Cube: `44ee834999e81d01974119326efa4c9b722ea0b8b18cf715d7b797f056b7de57`.
- TwoRoom: `1d7d9ab93ac3f8f7328c7f6bf41cb576992513d9bdd19abff9b58bd41e64f84e`.

## Exact operation and native contracts

Original diagonal CEM operates on y. Only its cost wrapper maps x=Ly before the unchanged model/objective; its returned whole plan is mapped to x before the unchanged policy/scaler/execution/history. Initial y~N(0,I) gives x~N(0,R), preserving zero mean and unit individual coordinate variances. Later covariance is L diag(s_y²)Lᵀ. The unbounded feasible action set is unchanged because L is invertible; finite-budget search is intentionally different.

No raw action clipping or covariance-driven goal cost is introduced. The small linear map is FP64 with one cast to native action dtype, avoiding TF32 dependence; its extra arithmetic is reported separately from the unchanged learned-model evaluation budget. Original mode and exact L=I bypass all mapping, casting and tensor replacement.

Actual native settings remain300 samples×30 iterations/top30, model horizon5, raw action block5, real observation history3, receding horizon5 and raw evaluation budget50. Native `var` means standard deviation; logged mean/std remain explicitly labelled white coordinates in correlated mode. If converted, original-coordinate marginal std is sqrt((L**2)@(s_y**2)), never L@s_y. The native solver has no std stopping rule or early termination; all30 iterations remain. Because receding=horizon, the whole25-raw-step plan is consumed and there is no warm tail. The adapter rejects nonempty warmstart, actor costs and nonempty callbacks rather than silently interpreting unsupported coordinates.

Nine synthetic CPU contracts passed, with no learned models or simulator: both task dimensions, native/original/L=I identical30 candidate banks/costs, final moments/actions and RNG; nonidentity Cholesky/unit marginals, mapped first draw/forced zero/return; rejected partial warmstart; actual train scaler with50 returned commands/two replans and second-plan executed history in mapped coordinates; all hooks restored after deliberate exception. Native source and actual policy scaler are checked again on each real run. The first real plan records original imported model paths/hashes, finite actions/cost and CPU-vs-GPU mapping error; failure holds the run, never changes a matrix or silently falls back.

## Bounded48-endpoint allocation

Twelve already held checkpoints, each original/correlated and original primary/fresh50 cases:

- TwoRoom: F64 **unconstrained/free chart**6000, native LeWM **raw**6000, TF6000; seeds1/2.
- Cube: local1024 **identity**6000, native LeWM **raw**12000, TF12000; seeds1/2.

TwoRoom F64free and Cube local1024 are different existing candidates, not one unified algorithm. Native and candidate capacities differ; Cube local has6000 versus native/TF12000 exposure. These limitations survive a shared controller. All original model sources remain32fad0c/7a1ab4f/8e0f821, while the new helper snapshot is separately recorded. No BN-calibrated export is substituted.

Predeclared slot order is task(TwoRoom,Cube), family(listed above), seed(1,2), case(primary,fresh), mapped sequentially to dev2g0–7, dev13g0–7, dev15g0–7. Each slot executes original then correlated for exactly the same checkpoint/cases:24 slots,2 waves,48×50 episodes. All GPUs must actually be free before activation. The preserved candidate root stays held; a newly frozen official root gets a separate reviewed active-plan copy. All previous roots/holds remain unchanged.

Original success vectors are recomputed and compared with historical references. Known Cube reset variability means outcome-bitwise replay is not required, and the best of old/replay/bare runs is never selected. Main paired comparison uses the new original arm; historical references remain separate. Read all48 results before outcome comparison; report all seeds/arms/cases, case-paired uncertainty conditional on those seeds, and note that bootstrap excludes environment-repeat variance. No automatic follow-up.

## Established prior and boundary

[iCEM, Pinneri et al.](https://proceedings.mlr.press/v155/pinneri21a.html) already uses temporally colored action noise, along with other search modifications. This control instead isolates a fixed training-covariance coordinate system with unchanged native diagonal CEM; it is neither an originality claim nor a reproduction of complete iCEM. Matching empirical action support does not establish optimality: zero commands are highly central but fail nearly all Cube cases, and several within-family success/support comparisons have the opposite direction. The control can fail despite reducing spectral mismatch. No outcomes may be used to retune it.
