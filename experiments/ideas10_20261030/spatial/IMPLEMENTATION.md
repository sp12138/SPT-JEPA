# Optional response-query prototype

Implemented only in `as_jepa_lewm/robust.py`; contracts in `tests/as_jepa/test_response_queries.py`. The parent owns GPU preflight, source freezing, and launch decisions. This file describes the exact implementation, not completed GPU validation.

## Unchanged interface and paired modes

`response_query_mode="none"` is the default and creates no module or state-dict entry. Its loss, metrics, gradients and RNG must be bit-identical to frozen `b3f27e6`. The existing `RobustLeWM` loader, Trainer, exporter and checkpoint mechanism remain in use. No new subclass or dispatch path is introduced.

`direct` and `annihilator` add exactly the same `Linear(state_dim,192)` head. At the current64-dimensional state this is12,480 parameters, two extra tensors, approximately0.0244% of the51,097,134-parameter baseline. Construction occurs inside an isolated RNG context after all baseline modules. All550 old initial tensors and the caller's RNG must remain unchanged; direct and annihilator have equal head initialization for a given seed.

Both modes roll the existing predictor forward from the clean initial encoded state through the seven existing action blocks. The future observed states are not supplied to this rollout. The head reads all eight resulting states. “Clean” means no posterior sampling noise; training-mode predictor dropout is retained and uses an isolated stream `forecast_seed+1000003+step`. Native posterior, DiT, rate, anchor-image and corruption paths are unchanged. The extra seven predictor applications have a real compute cost despite the tiny parameter increase; count this when comparing against the old TF control.

## Exact target, projector and loss

The target ψ is `adaptive_avg_pool2d(render,8×8)`, flattened to192 RGB values in the existing[-1,1] scale. This prototype uses simple area-pooled pixels, not the smooth cosine measurements used in the preliminary CPU data audit. The latter's Taylor-order example is a conditional mechanism test; it does not establish smoothness of all quantized rendered pixels or hard cell boundaries in this prototype.

For an eight-frame clip, form cumulative sums of its standardized controls, after summing each five-raw-action block. Ignore the final padding action. Define

\[
X=[\mathbf1,t,U],\quad P_{\rm direct}=I_8,\quad
P_{\rm ann}=I_8-XX^\dagger.
\]

The time column ensures raw↔standardized action centering changes only the span: cumulative raw control equals cumulative standardized control times its scale, plus a time-linear mean term. No dataset means/stds, physics labels or extra observations enter the model. The block size5 is an explicit existing benchmark data contract; other block sizes are not automatically supported.

Columns are normalized before a detached FP64 SVD, preserving their span while stabilizing units. Numerical rank uses `max(rows,columns) * eps(float32) * largest_singular_value`, reflecting the stored action precision. The projector is cast toFP32 for the small residual multiplication. Let e_t=predicted ψ_t−observed ψ_t. With a fixed auxiliary coefficient1,

\[
L=L_{\rm native}+\frac1B\sum_b
\left[\frac{\|e_{b,0}\|^2}{192}
+\frac{\|P_b e_b\|_F^2}{192\operatorname{rank}(P_b)}\right].
\]

The projected term is zero only when its computed rank is zero; the shared grounding term remains. Direct has rank8. **Direct intentionally gives the initial frame total coefficient1+1/8 and each future frame1/8.** Annihilator includes the same explicit initial grounding term plus the initial frame's contribution to the temporal contrast. This is not hidden double counting: the two arms share the exact loss family and differ only inP.

The per-clip projection couples frames of one sample, never distinct samples. The fixed target has no gradient. Base native objectives remain intact; no proposed batch-level penalty is added. At inference the existing encode/predict/GoalMSE planner is unchanged, and the auxiliary head is unused.

## Rank and diagnostic limitations

Existing128-window training-data check: TwoRoom rank(P)=4 for128/128. Cube rank(P)=1 for127/128 and2 for1/128. Thus the Cube prototype retains only one temporal contrast for almost every sampled window. No sampled window has rank zero. This is a material limitation, not a numerical failure.

The existing frozen information audit's `flow` and `priced_rate` gradient diagnostics still refer to those original components; they do not suddenly include the auxiliary gradient. Full-loss health checks and the new CPU contracts must establish auxiliary gradients separately. Renderer-condition sensitivity remains a different quantity from clean latent planner sensitivity.

## Required validation before600

CPU contracts cover default exact regression; old tensor/RNG preservation; equal head initialization and parameter counts; strict state-dict reload; projector rank, symmetry, idempotence, affine action scaling/centering invariance, padding independence and sample independence; rank-zero finite behavior; an independent least-squares oracle for the exact loss; causal rollout/no future-state leakage; nonzero finite encoder/pool/predictor/head gradients; and native metrics/outer RNG preservation when the auxiliary path is active.

Real two-step preflight, checkpoint/export equality, full gradients and all short-gate scheduling remain parent-controlled. Finite CPU or two-step behavior is not a functional600-step result and does not authorize longer training.
