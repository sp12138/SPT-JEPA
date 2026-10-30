# Temporal targets: a concrete repair that does not pass our current data gate

2026-10-31. The [TD-WorldModel preliminary report](https://td-worldmodel.github.io/TD-WorldModel/) trains unit-norm predictions and frame embeddings against within-clip cosine targets `τij=exp(−|i−j|/σ)`. It reports a single Scene-environment study and emphasizes that richer representations trained only on expert data extrapolate poorly to changed actions. Its Scene results are not our Cube results; its reported checkpoint selection and augmented data must remain visible in any comparison.

## Exact feasibility and a counterexample

Writing ρ=exp(−1/σ), the target matrix `τij=ρ^{|i−j|}` is positive definite for 0<ρ<1. One constructive realization is the unit-vector AR(1) recursion `v_i=ρv_{i−1}+sqrt(1−ρ²)e_i`, with each fresh e_i orthogonal to preceding vectors. For an n-frame clip the target has rank n, so exact realization requires embedding dimension at least n, as well as consistency with a shared observation encoder. The product relation between successive temporal targets alone is not a complete realizability argument.

If exactly the same image occurs at i and k, its deterministic encoded vectors coincide. For any prediction of target i, the two cosines against z_i and z_k must therefore be equal. Their desired values are 1 and ρ^{|i−k|}. The minimum sum of their two squared errors is

\[
\frac12(1-\rho^{|i-k|})^2>0.
\]

This is a direct contradiction between discriminating time and respecting observation identity. Conversely, if all encoded frames collapse to c but unit-norm predictions remain flexible, the optimal cosine in row i is the mean target in that row; the residual loss is its target variance. Thus collapse cannot achieve zero, but noncollapse does not ensure physical information: a predictable visible clock could provide temporal discrimination without encoding the controlled configuration.

## A specific target update

Let S assign each frame to its exact-image equivalence class. Form

\[
G=S^\top\tau S,\qquad D=\operatorname{diag}(G),\qquad
K=S D^{-1/2}G D^{-1/2}S^\top.
\]

Replace τ by K in the same sample-level cosine regression. K is positive semidefinite, has unit diagonal, and identical observations receive identical rows with mutual target one. When every observation is distinct it reduces exactly to τ. This is a different update rule on repeated-image clips; the algebra is ordinary normalized kernel-mean construction, not a new theorem.

It does **not** fix global consistency. The same two distinct observations A and B can occur one step apart in one trajectory and four steps apart in another with no within-clip duplicates. With ρ=.5 their targets remain .5 and .0625. A common unit-norm state geometry cannot satisfy both exactly alongside perfect diagonal prediction. Turning this into a global graph metric would be a different method with substantial existing literature and behavior-policy dependence.

## Current actual-data result: no useful distinction

The CPU check sampled 128 training and 128 disjoint validation episodes per task, one 8-frame benchmark window each. It used exact raw-image byte equality, a sufficient condition for equal encoder observations, with no thresholds or physical labels.

| Task | Split | Clips with any repeated image | Mean distinct images per clip |
|---|---|---:|---:|
| TwoRoom | train | 0/128 | 8 |
| TwoRoom | validation | 0/128 | 8 |
| Cube | train | 0/128 | 8 |
| Cube | validation | 0/128 | 8 |

Consequently, the proposed quotient target is exactly ordinary TDWM on all 512 sampled clips. Enlarging the equivalence relation with a tuned image/latent threshold would introduce a new assumption and could identify physically distinct states; a learned equivalence can also reintroduce collapse. We do not add that modification to make the candidate pass.

**Decision:** keep the counterexample and target construction as a theoretical control, not an admitted new hard-task method. The algebraic checks pass, including the unresolved different-path contradiction. No GPU or model training was run.

Artifacts: [check_temporal_quotient.py](check_temporal_quotient.py), [CPU outputs and episode records](ideas10-temporal-quotient-20261031.json).
