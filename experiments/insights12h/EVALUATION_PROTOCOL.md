# Evaluation protocol clarification

The first L0 P0 result (39/50 success at 25,000 updates) uses the pinned official script's default sampling from the complete expert-training file. It is not an episode-disjoint result. Its case seed is 42; the upstream script does not explicitly seed torch before the CEM loop. Treat it as a preliminary protocol control.

The repository-owned planner entry still runs that pinned upstream script. An optional held-out restriction changes only selection lengths for non-test episodes to zero, which makes those starts invalid under the official goal-offset filter. It leaves source pixels, reset states, global action normalization, CEM, costs, history and budget unchanged. It records selected episode IDs and start steps, asserts test membership, and seeds Python/NumPy/torch immediately before World.evaluate. The default precision is FP32. Do not pool full-file and held-out results.

Counterfactual evaluations use the same initial simulator reset for four donor action sequences and the same sampling noise for every donor. Report both the learned model's assignment accuracy and a pose oracle operating on real simulator frames. Report the recorded-action replay error because dataset states do not save block velocity and exact simulator restoration is not possible from those states. Counterfactual action identification is distinct from P0 planning success.

The frozen pose evaluator's real-frame error is measured separately on standard, episode-color and random-dot domains. Training loss, posterior mean decodability, noisy-code decodability, open-loop image/pose error and planning are separate measurements.
