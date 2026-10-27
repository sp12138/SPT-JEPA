At step 2000, standard-data L2 at beta 1 through 256, beta=0 and the unpriced L1 all fail to beat copying the last context frame. State standard deviation is about 0.0007, versus sigma0=0.1; zeroing the state barely changes the renderer. This does not isolate a price-induced collapse. Two focused diagnostics are added: (1) a one-time affine head initialization on 128 training windows targets mean per-dimension std 0.1, changing only initial parameters, with no running normalization; (2) the documented L1-only BatchNorm fallback. They are labeled separately from the canonical initialization. Early renderings still contain sampling noise, so no final collapse claim is made from step 2000.

## Encoder warm-start diagnostic

A separate diagnostic initializes only the ViT encoder from this campaign's 25k L0 checkpoint. Pooling, action encoder, prior, renderer and optimizer remain fresh; the AS-JEPA objective is unchanged. One-time head calibration and an initial portable export are recorded. This tests the joint-optimization barrier and must not be reported as learning from scratch or as a matched compute comparison. Full resume skips both encoder initialization and calibration.

## Numerical precision controls

Matched frozen-weight probes show that several amplified-initialization checkpoints have BF16 feature std around .15-.2 while full FP32 std is only roughly .0001-.0004. Their noisy FP32 state probes remain near chance. Thus high BF16 variance in these arms contains substantial numerical variation; it is not evidence of useful retained information. Normal L1+BN and beta0 controls have similar BF16/FP32 position scores.

Two additional beta=.25/std=1 controls keep the same mathematical model/loss, batch128 and data: BF16 encoder with an FP32 QueryPool, and full FP32 training. Both use float32 matmul precision=highest. Record actual speed/memory before committing a long budget; a successful FP32 inference probe alone does not prove FP32 training will prevent optimization collapse.

## Staged optimization diagnostic

The encoder-only warm initial model has noisy-code position R² .516/.792 on fixed test frames, but after 4k joint updates its beta=.25 version falls to .125/.153 and beta=4 approaches zero. A separate diagnostic freezes encoder and pool optimizer updates for the first 4k steps while prior and renderer learn, then resumes joint updates. This is explicitly a staged-training ablation; it is not the canonical from-scratch, end-to-end method. Lightning's existing optimizer hook and complete checkpoints are used, with no new Trainer/checkpoint implementation. Verify the frozen weights remain exactly equal to the initial export during the first 200-step smoke.
