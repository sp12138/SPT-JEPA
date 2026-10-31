# One bounded design attempt: amplify small changes by fixed error-detecting codes

Design recorded before the focused primary-literature check; no GPU/model/data reads.

Motivation: area-averaged image prediction can allocate a bottleneck to a repeated predictable background or large arm rather than a small independent object. Prior addressed-query/mixed routes improved observed information without useful causal forecast; hybrid protected a fixed observation carrier but not its prediction; patch2000 did not improve the fixed target dynamics. A new auxiliary head is therefore not evidence of improvement.

The candidate replaces an image-area target by a fixed, object-agnostic binary code. Quantize RGB with a fixed public rule, write its bits as x in GF(2)^D, fix iid binary rows R using the declared seed, and define c_j(x)=(-1)^(R_j x). The observation state is this fixed code; the original action-conditioned predictor forecasts it directly under sample-level MSE, and its observed goal uses the same code/Euclidean cost. No physical masks, poses, optimal actions, other examples or pretrained encoder enter the rule. Its intended advantage is that one small changed region flips an O(1) fraction of output bits rather than an area-proportional fraction.

This is deliberately one mechanism, not a proposed experiment: first test whether it preserves useful conditional prediction and graded goal geometry. Prove the effective kernel, compare one-bit capacity allocation with replicated background, and construct an invertible small-change-sensitive code whose Bayes causal forecast loses object dependence under ordinary unpredictable observation noise. A fatal counterexample rejects it; no attempt to rescue it by masks, tunable sparsity, extra heads or a second candidate.
