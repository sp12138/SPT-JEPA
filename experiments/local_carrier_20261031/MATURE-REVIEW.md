# Completed bounded stages: prediction improvements are not control improvements

No tested sample-level method has yet shown substantial superiority over the appropriate baselines on both tasks. At6000 updates, the local1024 carrier improves some Cube comparisons but remains weak in absolute performance and trails the stronger64-D carriers/native LeWM on TwoRoom. Neither greater state/action sensitivity nor lower factual prediction error is sufficient evidence for better planning.

|6000-update model|TwoRoom primary / fresh, seeds1/2|Cube primary / fresh, seeds1/2|
|---|---|---|
|TF64|2/4 ;4/2|24/14 ;14/18|
|Fixed64 identity|80/86 ;74/80|10/12 ;12/10|
|Fixed64 learned free chart|90/94 ;82/86|Not continued to6000|
|Fixed64 learned bounded chart|92/88 ;80/94|Not continued to6000|
|Local1024|60/60 ;72/60|16/22 ;24/20|
|Dense1024|12/14 ;20/18|12/18 ;10/14|
|Native LeWM, raw|80/72 ;76/70|24/22 ;12/14|
|Native LeWM, fixed train-only BN calibration|82/78 ;84/88|16/18 ;20/16|

Entries are success percentages, with both training seeds retained. Data/update exposure, ordered cases, planner budgets, action normalization and strict success semantics are matched within each comparison. Native LeWM is approximately18M parameters, F/TF51M and local/dense56M; this is not equal capacity or compute. Baselines retain their native objectives; only our candidates are required to use sample-level losses. Fresh means new goal pairs from the existing validation pool, not an independent test dataset. We do not choose raw versus calibrated LeWM separately by task or case set.

Cube local−TF6000 is0pp[-7,7] on primary and+6pp[-1,14] on fresh; local−calibrated-native is+2pp[-5.025,10] and+4pp[-7,15]. These paired95% intervals condition on two fixed training seeds and do not establish a stable advantage. Meanwhile local causal rollout/copy MSE falls from.361/.354 at2000 to.227/.221 at6000, and DiT wrong-state sensitivity grows from1.57/1.68 to8.47/8.39. This is learned input use with limited task progress, not persistent collapse. Extending Cube TF from6000 to12000 likewise improves denoising diagnostics but changes success by0pp[-8,8] primary and−1pp[-9,7] fresh. Those training branches are held.

The orthogonal control supplies a cleaner mechanism counterexample. Its fixed Q preserves observation information and Euclidean goal distances to rounding. At2000 it has lower factual rollout and one-step errors in all four task/seed pairs, yet TwoRoom planning falls16pp[-24,−8] primary and21pp[-33,−10] fresh. Therefore neither retained information nor factual average MSE alone explains planning success. This does not identify the responsible network component: Q changes predictor and renderer parameterization, and independently trained predictors are not forced to be conjugate. Errors under planner-proposed actions and how those errors change candidate ranking are still unmeasured.

![Orthogonal-control prediction and planning results](results/mechanism-summary/basis2000-mechanism.png)

The next bounded work tests three separate questions, not three claimed innovations: a shared invertible16-channel chart asks whether locally learned coordinates help without discarding detail; a frozen decoder-response metric tests the mismatch between generation geometry and planning distance; a common-action-bank environment audit tests factual-versus-counterfactual ranking accuracy. Each starts with CPU contracts and a committed-source technical/short gate. No new long grid or automatic continuation follows from this review. Decoder geometry is a controller diagnostic and the common-bank rollouts supply diagnostic labels only, never additional training data.

Full supporting reports: [local6000](results/gate6000/RESULTS.md), [orthogonal2000](results/basis2000/RESULTS.md), [native6000](../native_baseline_20261031/results/gate6000/RESULTS.md), [TF12000](../carrier_progress_20261031/results/tf12000/RESULTS.md). All relevant stages completed successfully, original checkpoints and historical holds are preserved, and idle owned dispatchers are closed. The high-performance goal and the request for ten genuinely distinct evidence-backed successful ideas remain open.
