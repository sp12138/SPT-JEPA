# Authorized stage:8 native two-update runs +8 strict fullstate audits

Root explicitly authorized production integration and ONLY this technical stage.
No600 plan is created. Reserved slots:dev2 GPUs0–2,dev13 GPUs0–2,dev15 GPUs0–1.
The16 endpoints use the original native Trainer, two optimizer updates at
batch128, max_steps124910/warmup1249 and original immutable local1024 train-only
calibration/data. Memory failures remain failures; no silent batch/width/clip
change. Historical holds and all other slots are untouched.

The new `PatchCarrierLeWM` inherits the exact original `encode`, `prepare`,
`predict` and `RobustLeWM.loss`; it changes only the predictor constructed after
the original model. Window8, local1024, carryTrue and no chart/readout combination
are required. Explicit `patch_predictor_seed=training_seed` gives paired
identity/orthogonal parameters and independent training-seed initialization.
Private initialization uses the CPU default generator only; it does not call
CUDA seeding APIs. The fixed Q lives only inside prediction and is inverted on
the residual before native carry. Raw goal, renderer conditions and rate remain
in the same original coordinates.

Final CPU v2 verified all466 original nonpredictor tensors against the four
actual archived initial models, exact outer initialization RNG, all542
parameter tensors paired (including frozen parameters), and different predictor hashes for seeds1/2. The CPU
JSON field `paired_learned_tensors` counts all named parameters, not only trainable ones. Default
production model/training sources are byte-identical; factory only adds one new
explicit target. Full model totals are45,047,806/45,109,786 for TwoRoom/Cube,
predictor523,536. Original native flat predictor11,389,184 is removed.

Two actual native AdamW/scheduler updates in a synthetic B1,window8 CPU fixture
(smaller encoder/DiT only for that loss test) open input/condition/output,
position, layers, action encoder and renderer gradient routes. Model, optimizer
and scheduler round trips are exact. The two token arms have identical initial
loss/metrics/runtime RNG and49 nonpredictor gradient tensors in each task.
Actual production-sized initialization and production-class frame-mask/native
cached H1/H3 rollout tests separately pass. These do not establish real batch128
CUDA memory or learning health. Old CPU v1 evidence is retained; v2 fixes the
private CUDA-seeding side effect before any GPU source is frozen.

Flat and token architectures consume different runtime dropout tensor sizes.
Only their original distributions and loss/data are matched; no claim of equal
individual runtime noise draws and no RNG reordering patch is introduced.

Each actual two-update checkpoint first receives the unchanged original strict
native preflight (one FP32 loss backward, no update), including finite fullstate,
optimizer/scheduler/RNG/cursor, exact portable weights, frozen encoder/pool/
calibration and live module gradients. The owned wrapper adds checks of every
patch layer's attention/MLP/AdaLN and input/condition/output/position gradients
from that SAME backward; it adds no backward or relaxed expectation. CPU checks
then verify all466 initial nonpredictor tensors against old local, all paired
parameters and common buffers, and actual native optimizer/scheduler restoration
with exact trainable membership, no frozen state and native upcoming sampler IDs.
The original report and new sidecar both remain authoritative, with hashes.

After all16 succeed, review source/config/calibration, per-seed initialization,
memory and actual fullstate before any further decision. The proposed600 gate
is still unauthorised. A useful future outcome must concern paired action-use,
multistep predictions and actual primary/fresh planning, not only rank/parameter
count or factual loss. This is a DINO-WM-related architecture control, not a
claim of a new learning mechanism.
