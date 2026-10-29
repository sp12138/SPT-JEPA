# Prior energy for within-video action ranking

For a video window x and candidate action sequences a^(0),a^(1),a^(2) (observed, reversed, shifted), use existing latent energy

    E_k = mean ||F(encode(x),a^(k)) - encode(x_future)||^2 / sigma0^2
    L_rank(x) = -log softmax(-E)_0.

The one-step variant uses teacher-forced history. The terminal variant rolls five steps from the first three states and uses the final latent, matching the P0 horizon/history. No new classifier or parameter is introduced. All encoder inputs/targets remain differentiable. The original sample-level flow/rate losses remain intact.

Alternatives come only from the same sample; a minibatch averages independent L(x). They are not simulator-confirmed negative outcomes: repeated or causally equivalent actions may reach the same goal. If allcandidateactionsareidentical, private common dropout gives identical energies, log3loss and no candidate-index learning signal; the contract checks this in training mode. Common outer RNG state is preserved for the original posterior/renderer estimator.

Two fixed auxiliary strengths (1e-4 and1e-3) expose sensitivity without claiming an optimum. Train/evaluate allfourpairedexistingseeds for600updates and stop. Improving this ranking is insufficient evidence of planning improvement; evaluate actual P0 and independent action cases.

The underlying contrastive/action-conditioning principles are established research (CAPE, TWISTER). This loss is a sample-local adaptation and is not claimed groundbreaking. Extra prior calls consume compute, so report cost alongside parameter/update count.
