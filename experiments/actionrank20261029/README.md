# Sample-level action ranking

Five settings including original control x four paired existing2epochtraining seeds, each600updates before review. Teacher-forced1step and deterministic5stependpoint rankings each use two fixed coefficients. Alternatives are reversed/shifted actions from the same video; no cross-sample negatives, extra head, state/reward labels or new parameters. All original flow/rate gradients retained. Common private dropout per candidate prevents candidate-slot leakage.

This is a contrastive surrogate, not simulator-proven counterfactual training. Alternative actions may have the same outcome. It is related to known action-contrastive world-model research; no novelty claim. Short adaptation does not prove from-scratch stability.

Related primary sources: [CAPE](https://arxiv.org/abs/2606.07304), [TWISTER](https://arxiv.org/abs/2503.04416). These motivate action-conditioned discrimination; our sample-local loss is an exploratory adaptation, not a reproduction. Prior I09 showed that fitting an auxiliary inverse classifier can destroy planning, so this screen uses the existing prior/goal energy and retains original grounding gradients.
