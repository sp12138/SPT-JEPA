# Evaluation protocol clarification

The first L0 P0 result (39/50 success at 25,000 updates) uses the pinned official script's default sampling from the complete expert-training file. It is not an episode-disjoint result. Its case seed is 42; the upstream script does not explicitly seed torch before the CEM loop. Treat it as a preliminary protocol control.

The repository-owned planner entry still runs that pinned upstream script. An optional held-out restriction changes only selection lengths for non-test episodes to zero, which makes those starts invalid under the official goal-offset filter. It leaves source pixels, reset states, global action normalization, CEM, costs, history and budget unchanged. It records selected episode IDs and start steps, asserts test membership, and seeds Python/NumPy/torch immediately before World.evaluate. The default precision is FP32. Do not pool full-file and held-out results.

Counterfactual evaluations use the same initial simulator reset for four donor action sequences and the same sampling noise for every donor. Report both the learned model's assignment accuracy and a pose oracle operating on real simulator frames. Report the recorded-action replay error because dataset states do not save block velocity and exact simulator restoration is not possible from those states. Counterfactual action identification is distinct from P0 planning success.

The frozen pose evaluator's real-frame error is measured separately on standard, episode-color and random-dot domains. Training loss, posterior mean decodability, noisy-code decodability, open-loop image/pose error and planning are separate measurements.

One default-initialization beta4 arm is deliberately continued to 25k after the initial 8k screen. Its STOP_DISABLED marker is declared before launch; the evaluator records a triggered collapse gate but does not stop that single control. This audits whether stopping at 8k was premature. Other screened arms remain stopped.

New evaluator snapshots additionally compare a noisy posterior state with the deterministic mean using identical renderer noise. Existing metrics directories retain their original protocol and are never overwritten.

The early streaming S0/S1 pixel evaluations used gap=0 in window enumeration while latent-model evaluations used gap=8; consequently those early pixel baselines did not share the same case IDs. Corrected evaluations use gap=8 for all models, while S0/S1 still receive their configured first-frame anchor. Final output directories must use the corrected snapshot and retain cases.json; do not claim paired case comparisons for the older streaming measurements.

The planner wrapper passes the pinned script's absolute Hydra config path. Loading that script through a named module otherwise makes Hydra infer a nonexistent '..config' package. Failed startup batches are retained as failures and excluded from results.

The v1 pose evaluator fails its random-dot real-frame control (10k test frames: agent 108.41 px, block 29.57 px). Dot-domain pose scores from that evaluator are excluded. A v2 evaluator is separately fine-tuned from v1 with independent random-dot augmentation (half of training steps), using training pose labels only and a fresh optimizer/schedule. Its standard, color and dot real-frame floors must pass before final generated-image pose comparisons. Keep v1 weights and outputs unchanged.

Mean-code probes additionally report the tighter full-covariance Gaussian-noise channel bound, alongside the diagonal bound. Correlated dimensions can make the diagonal bound loose; neither bound establishes task information. The batch probe entry reuses process imports and backend initialization only; each probe resets its declared seed and reloads its immutable model.

Native exploratory runs use one AdamW group (lr=1e-4, betas=.9/.95, weight decay=0) and a constant schedule after warmup. This differs from the LeWM encoder/prior and renderer groups and cosine schedule. Consequently Native-vs-LeWM differences cannot be attributed solely to architecture. Native S2 was stopped after verifying its 12k full DCP because its noisy state remained empty; its task has no STOP callback, so only that owned process group was terminated and its unsaved tail was discarded.

Gradient diagnostics compare flow and priced-rate derivatives on the same fixed train minibatch without any optimizer update. A zero initial flow-to-state gradient is expected from a zero renderer output head; its presence alone is not evidence that later learning is impossible.

A matched FP32-versus-BF16 feature probe is added for the same frozen weights and case IDs. The large one-time head gain can in principle amplify rounding/cancellation; this is a diagnostic hypothesis, not an established cause. Feature extraction precision and float32 matmul precision are recorded. Do not infer useful state information from variance or an information upper bound alone.

Full-FP32 training controls are evaluated with FP32 forward passes explicitly, and mixed controls retain BF16 autocast with the declared FP32 pool where applicable. The streaming evaluator's next revision uses pose-v2 and a separate eval-posev2 directory; it preserves all earlier eval directories. The pose weight path and feature-forward precision are stored in each new measurement.
