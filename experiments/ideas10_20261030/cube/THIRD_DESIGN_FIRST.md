# Third design pass: terminal controlled consequences and finite-noise distinction

2026-10-30. Written before the targeted prior search. No claim of novelty or efficacy.

## D7: deepest nonconstant controllability-order output

A universal semantic goal quotient is unidentified, but an explicit intrinsic preference might still distinguish a sole controlled agent from an actuator mediating object motion. In control-affine dynamics dx/dt=f(x)+sum_i g_i(x)u_i, an observation function h has relative degree r if L_gi L_f^j h=0 for every i,j<r-1 and at least one L_gi L_f^(r-1)h is nonzero. Prefer the deepest *reachable, nonconstant* output, not every invariant output. Directly controlled navigation has no nonconstant deeper output, so its position remains. A chain u→actuator→object admits an output that ignores immediate actuator perturbation yet responds after mediation.

Learning version: preserve a full prediction state, but fit a hierarchy of output heads from matched bounded action pulses and their genuinely observed future response. Head r suppresses earlier response contrasts and must predict a nonzero later addressed-observation contrast. The proposal's goal discrepancy uses the deepest head with held-out controlled-response evidence. No physical object or task-success labels identify the head. All branch data and inference overhead go to matched baselines.

This is not D4's quotient of all reversible movements: the nonconstant reachable layer requirement gives a fallback to the direct actuator when no deeper consequence exists. It is not a Gramian inverse; the preference is ordinal delay/dependency, not energy normalization.

However relative degree is classical. The possible contribution is the explicit reward-free preference for terminal controllable consequences plus its learned observation-defined filtration. It is an inductive bias, not recovery of arbitrary user semantics. A navigation task with a controllable door can make the door deeper than the agent; a task whose goal is robot posture makes the same preference wrong. Arbitrary mixing of observables, hidden controller states, sample rate and sensor noise can obscure response order. Learning a global causal decomposition from passive data is not guaranteed.

## D8: a statistical distinguishability contract for the noisy state channel

Two current states can have useful clean probes but become nearly indistinguishable after fixed Gaussian posterior noise. For q_i=N(mu_i,sigma²I), their total variation is 2 Phi(||mu_1-mu_2||/(2 sigma))-1 and optimal equal-prior discrimination error is Phi(-||mu_1-mu_2||/(2 sigma)). Under an action-conditioned stochastic decoder with identical side information, data processing requires TV(q_1,q_2)≥TV(P_future1,P_future2) if it represents both outcome distributions exactly.

Proposed learning rule: create matched same-action pairs with identical permitted side information, estimate a lower confidence bound on a fixed observation test's future distinguishability, and impose the corresponding minimum separation through the actual noisy latent channel. A pair is one sample-level unit; no batch covariance/negatives. A finite tolerance is necessary: deterministic distinguishable observations have TV=1 and would demand infinite Gaussian mean separation for exact domination.

For allowed state discrimination error eta, separation at least 2 sigma Phi^{-1}(1-eta) is necessary. Under a standard-normal entry prior, the minimum average mean-dependent KL cost of two symmetric Gaussian codes is distance²/8. This yields an explicit capacity price for preserving one distinction, not just a raw variance target.

Limits: with deterministic outcomes and a fixed eta, the objective reduces to a contrastive distance margin, potentially an already-known method in different language. Shared action labels are not enough if bypass/anchor differs. A renderer also given the noised target has extra side information, so the naive state→future DPI does NOT apply to its conditional denoising graph. The contract applies to the full generative/planning channel under fixed common side information, not arbitrary FM training pairs. Unknown physical variables and future semantics cannot be recovered by the inequality alone.
