# One fixed full1024 orthogonal-basis control

**Diagnostic only; ready for review, no training or GPU launch.** Existing local/dense changes spatial averaging, projection subspace and coordinate grouping together. This control retains the local measurement and applies one full orthogonal map **after** its existing global-RMS normalization. It asks whether the useful coordinate arrangement matters to finite-budget learning/conditioning even when information and the ideal goal metric are preserved. It is not a new algorithm or novelty claim.

## Exact construction and implementation

For the unchanged normalized local code `z=(r−mean)/scale`, use `z'=zQ`. Q is a1024×1024 signed/permuted Sylvester Hadamard matrix divided by32, constructed by a fixed private NumPy stream `[20261031,1024]`. Every output mixes all1024 inputs. Q is the same for both tasks and training seeds; it is not a Haar-random rotation and only this one basis is proposed. Orthogonal reflection versus proper rotation is immaterial to the preserved quantities.

`SpatialCarrierLeWM(spatial_basis='identity')` is the exact legacy default: no extra state-dict key, no tensor transform and no RNG consumption. `spatial_basis='orthogonal'` adds one persistent, nontrainable4MiB buffer `spatial_orthogonal_basis`. The learned model, parameter counts and initialization streams stay unchanged. Multiplication uses FP64 accumulation and casts back to the input state dtype (normally FP32), making this diagnostic independent of TF32 approximation in the *new transform*. Existing frozen carrier/train/eval numerical behavior is otherwise unchanged. GPU memory/latency remains unmeasured.

The transform is inside `encode`, so posterior means, observed future targets, initial planner states and encoded goals use the same basis. No extra head/loss, image, calibration, privileged label or action target is introduced. Recorded actions and native sample-level TF losses are untouched. `raw_carrier` remains the original pre-normalization measurement; do not recalibrate or rotate it separately.

## What is and is not invariant

Since `QᵀQ=I`, the real-arithmetic map is invertible, `z=z'Qᵀ`, and

`||zQ−gQ||² = ||z−g||²`.

Thus all observational distinctions and Euclidean observed-goal distances are retained. The scalar global-RMS normalization is already fixed; applying Q after it introduces no empirical coordinate reweighting. Finite FP32 output rounding means bytewise information preservation is not claimed.

An isotropic Gaussian channel is equivariant **in distribution**. If `z_noisy=μ+σξ`, its rotated coupling is `μQ+σξQ`; standard Gaussian ξQ has the same distribution as ξ. The native AR posterior with common scalarρ is also invariant because its time covariance has blocks `ρ^|i−j| I`, which commute with Q. The same Torch seed across arms instead draws the same ξ, giving `μQ+σξ`, generally not the same realization as `(μ+σξ)Q`. Do not claim pathwise loss/gradient equality between the rotated and unrotated arms.

The native entry/surprise rates retain their norms and constants when **posterior, target and predicted state are jointly rotated**. For a predictor F, the corresponding function is `F_Q(u,a)=F(uQᵀ,a)Q`; for a renderer D it is `D_Q(u)=D(uQᵀ)`. This does not say the native networks with the *same initial weights* implement these conjugated functions, nor that their sample loss is initially identical. Slotwise renderer conditioning, predictor parameterization, coordinate nonlinearities, initialization and optimization can all make the basis matter. It is therefore not a pure predictor/dynamics attribution.

The ideal planner metric is unchanged, but learned predicted-goal costs need not match because independently trained predictors need not be conjugate. Coordinatewise diagnostics (standard deviations, active-dimension counts and coordinate-standardized ridge fitting) are not orthogonally invariant and must not be mistaken for total-information measures.

## Falsifiable interpretation and proposed scope

A substantial rotated-versus-original local gap would show sensitivity to coordinate arrangement/finite learning or conditioning despite preserved measurement information and ideal Euclidean geometry. It would not identify which network component causes the effect, prove information inaccessibility, or establish a general rotation theorem. A small gap would show this one fixed mixing fails to explain the local/dense gap; it would not prove that pooling/information is its only cause.

If root approves a subsequent gate, the minimal proposal is four **fresh** rotated local runs (TwoRoom/Cube × seeds1/2) to600, using existing exact-compatible identity-local controls. Keep original global-RMS calibration, native hyperparameters, data order, planner budget and evaluation cases. First run the ordinary real2-step/gradient/memory gate. Explicitly add `spatial_orthogonal_basis` to the preflight's `initial_unchanged_prefixes`; all fixed buffers must remain equal to initialization. The old continuation helper enumerates the three original buffers and must not be silently reused for a rotated continuation without reviewing the extra buffer. Nothing here creates a source freeze, plan, calibration,600 launch or automatic continuation.

## CPU evidence

Four focused contracts passed in14.888s with CUDA disabled. The default regression used both tasks' **actual complete native configurations** (seed1, window8, batch1; no layer/width reduction) and verified source32 state keys/tensors, every metric, loss, every gradient and final RNG exactly. Rotated versus identity learned initialization/RNG also match. The actual1024 basis has zero FP64 orthogonality error, and a strict rotated-checkpoint encoding roundtrip is exact. Native rate tests jointly rotate the posterior/target/prediction and explicitly demonstrate the distinct same-noise-draw coupling.

A separate numerical contract uses only the four already-saved512×1024 local observed-state caches. It checks all130816 pairs per task/seed under the actual Q and FP32 return path, with a predeclared rounding-error bound. It performs no encoder call, label access, fitting or data-dependent Q selection. Small JSON/log results and EFS provenance accompany this protocol. Real GPU execution remains unverified.
