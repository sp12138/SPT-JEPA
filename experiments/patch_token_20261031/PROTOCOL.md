# Patch-token predictor: bounded CPU mechanism design

This directory is an independent prototype, not a production model/factory,
Trainer, launcher, checkpoint, or authorized GPU campaign. No datasets, images,
physical labels, trained model forwards, fitted probes, or optimizer updates are
needed for its CPU contracts. Native source is read from the exact currently used
`sources/stable-worldmodel` module/rollout files, and hashed before/after the run.

The single predeclared candidate uses existing fixed local1024 coordinates as
64 spatial tokens of16 channels; six blocks, hidden64, four heads of16 channels,
MLP256, dropout.1, native-style AdaLN gates and a shared16-channel output. The
existing1024-dimensional action encoder is retained. A shared Linear1024→64
projects its COMPLETE output once per frame; identical conditioning broadcasts
to all64 slots. AdaLN is computed once per frame. The original positional budget
`[1,7,1024]` is retained and reshaped to slots. Output/gates start zero; caller
adds unchanged raw carry. No alternative widths, heads or conditioning routes
will be tested in this design check.

The sole matched token control applies one fixed signed/permuted orthogonal
Hadamard transform to the state before the token predictor and its inverse to
the output residual. Action embeddings are identical in both arms. The raw
carrier, calibration, sample loss, DiT inputs, noise and goal metric are untouched
in this proposal. Parameter tensors/init are identical; the transform has no
learned parameters but costs two1024² FP64 matrix products per frame. This changes
the predictor's spatial prior, not the information available or raw goal L2.
It is not the old whole-model basis control and does not predict that either
basis will be better.

Attention permits `key_frame <= query_frame`, with ALL slots in the same frame
visible. Plain flattened-token causal masking is forbidden. State/action inputs
must have matching lengths; raw action history and future candidates follow the
actual native `LeWM.rollout`, including H1 and H3 contexts and default retained
history7. The outgoing action for frame t may affect its next-state output;
actions/observations after t may not. The contracts open otherwise-zero gates
and output in a synthetic fixture so that causality tests cannot pass vacuously.

The audit constructs the actual native global predictor (hidden192, depth6,
16 heads×64, MLP2048, input/output1024) and counts all parameters exactly. Linear
forward MAC formulas must equal hooks on actual CPU forwards. Dense QK/AV MACs
are explicit; softmax, nonlinear operations, encoder, retained action encoder,
DiT, backward and transfers are excluded. Native TF prediction calls have
lengths `[7,1,2,3,4,5,6,7]` at batch128. Native CEM has300 candidates×30 iterations,
horizon/receding5, action blocks5, history3, up to50 raw steps; predictor lengths
are1..5 with initial H1 and3..7 with H3, not a permanent3-frame truncation.

Execute once with2 CPU threads after writing source hashes. Only single B1,T7
forward/backward timings are collected per native/candidate to bound audit cost;
they are not GPU throughput or peak-memory estimates. Contracts cover private
RNG, parameter matching, zero-head carry, orthogonality/metric identity, full
frame mask, nonzero current-action/same-frame effects, zero future gradients,
padding rejection, cached native rollout alignment without encoder calls, and
shared-head local Jacobian rank. Stop after report; no GPU authorization follows.

The purpose is to remove a proven architectural output-space ceiling and test
spatial sharing. Patch-token prediction is already a known DINO-WM mechanism;
this is not counted as a new idea or a claim of novelty. A positive result would
still need actual task evidence and a disclosed compute/capacity comparison.
