# Active campaign recovery — checked 2026-10-27T16:43:52.903636+00:00

Deadline stays 2026-10-27 22:24:58 UTC / 2026-10-28 06:24:58 Shanghai. Authorized hosts: dev2, replacement dev13, dev15 (8 H200 each). Everything authoritative is /efs/prescottsun/SPT-JEPA. Do not touch other projects or restart old SPT historical jobs.

Campaign: runs/insights12h_20261027. Read plan.json, actual processes and worker-dev2/dev13/dev15.json. Training sources/configs are immutable Git archives; current code branch is insights/12h-20261027. No subagents have been used. New training is now closed; remaining work is verification, evidence analysis and final reporting, except repairs justified by actual failures.

28 finalizers are registered in finalization_registry.json. They run after their completed training parents on the same GPU. Standard runs get 20k/5k episode-disjoint probes, 256 fixed mean-rollout cases, 4 sampled latent rollouts/case where applicable, P0 seeds42/43/44 and 64 matched-reset action counterfactuals. Color runs get swaps; dot runs are compared on dynamics AND nuisance. Default source is source-final-v18 (7536848). The unstarted original-renderer finalizer was explicitly updated to source-final-v19 (49170c3), with 256 mean cases, 64x4 latent samples, planning/counterfactual priority. Read each finalizer actual_source.json/source_manifest.json rather than assuming the host worker source.

Long remaining training:
- L2-standard-calibrated-b001-50k-seed0: fresh 50k cosine budget, dev13 GPU3, expected finish around 20:15 UTC.
- L2-standard-original-renderer-b001-budget: 51,097,134 params, 384x12 renderer, dev13 GPU6; resumed trusted 200-step smoke; same 25k schedule, max_seconds=16200 (4.5h), expected endpoint around 20:35 UTC below 25k. Peak128.76 GiB, hot0.90s/step. Its final evaluation should finish before the original deadline.
- Full-FP32 beta.25 control: dev13 GPU7, ~0.74s/step, expected finish around19:15 UTC.
- Seed1 color anchor pair, L1 BN seed1, L1 MLP-BN, low-price dot pair, staged controls and beta1 controls remain in their explicit plan slots.

Source-cohort audit:
The initial source-v3 sampler advanced self.start during iterator construction. Corrected snapshots use completed-sample advancement, with exact cross-host CPU recovery tested. Legacy initial-grid recovery extensions are not claims of bitwise continuation of the old GPU/data order.
L0 legacy seed0 final P0 is 86/86/76 (mean82.7). Corrected-source seed1 is70/62/58 (mean63.3). A matched corrected-source L0 seed0 is now running at L0-standard-cursorfix-seed0, dev13 GPU2, with its finalizer queued. Matched corrected-source L1 without BN and raw beta4 seed0 were also added on dev15 GPUs6/4; beta4 preserves the full25k audit budget with STOP_DISABLED.

Current evidence (provisional):
- No AS-JEPA configuration has met the original three-seed generation gate. At256 cases, low-price beta.01/std.1 @18k mean h7 agent/copy=1.022 and moving-block/copy=.522. Small32/64-case screens were more optimistic.
- Legacy default beta4 extended to25k still has feature_std~.0005 and zero active dims. Beta1 extension @12k regained weak noisy position R2~.05, so an8k screen does not establish permanent irrecoverability. A separate uninterrupted beta1 seed1 (corrected source) is running.
- Amplified-head BF16 variance can be hundreds of times FP32 variance. Some noisy FP32 probes stay at chance despite large BF16 feature_std. Variance, effective rank and an information upper bound alone do not prove useful information.
- Color beta.25 @20k: with anchor, background-style transfer follows changed anchor (~.94) and scarcely changed state (~.001); without anchor, changed state transfers~.94. Renderings remain noisy and agent geometry imperfect.
- The no-anchor color state caught up substantially by20k. Paired episode-cluster bootstrap on5000 test frames/922 episodes gives anchor-minus-noanchor R2: agent [-.026,-.015], block [.047,.071]. Early10k advantage must not be presented as a persistent final accuracy improvement. Seed1 validation is still required.
- Warm encoder starts with useful noisy position R2~.52/.79, loses information under joint high-price optimization; staged freezing preserves207 encoder/pool tensors exactly during the200-step contract smoke. Staged controls are diagnostics, not canonical from-scratch results.
- Pose-v1 fails real random-dot frames (~108px agent); exclude its dot pose scores. Pose-v2 was separately trained and validated: standard/color/dot real agent errors2.37/3.03/3.25px, block1.48/1.80/1.78px.
- Nine CPU contracts passed on source-v14 including all903 inherited runtime hashes. CPU recovery5+15 vs20 steps has max parameter difference0; no bitwise GPU claim.

Read experiments/insights12h/EVALUATION_PROTOCOL.md and INITIALIZATION.md for departures. Source code contains as_jepa_tools/summarize_final.py: emits summary.json/CSV/RESULTS.md and plots, with episode-cluster bootstrap and explicit source-cohort metadata. Run its committed single-file export or a frozen archive remotely into a NEW review directory. Do not overwrite existing measurements. First partial review is review/final_partial_1620. Color paired-CI helper was exercised successfully on probe-batch6 data.

Still required:
1. Inspect every finalizer return code; repair actual failures with new immutable source/output if needed.
2. After W16 finishes, run supplementary latest-checkpoint probes (mean/noisy, FP32 where relevant) for W16/g4, direct prior/calibrated high-price, std.1 beta.25 and beta4, warm beta4, native S0/S1 representations and Native S2. Use pinned weights. Do not infer late outcomes from early probes.
3. Gather final 28-model evidence, compare corrected-source cohorts, two color seeds and both L0 seeds. Keep incomplete claims explicitly unresolved.
4. Inspect final visual galleries. Produce a concise Chinese insight report and small figures, leaving all checkpoints/logs/videos on EFS.
5. Stop owned workers/evaluator once all necessary work is complete (or let original deadline save/stop them); verify actual GPUs/processes. No compute extension past the original deadline.
6. Sync small source/report artifacts to the local review clone using Git; artifacts/ is ignored, so explicitly add only small chosen presentation files. The user explicitly authorized commit & push after main results are ready. The configured project origin is https://github.com/sp12138/SPT-JEPA.git (push dry-run succeeded); publish the insights/12h-20261027 branch, including code, result report and only small presentation artifacts. No repeat approval is needed.

If a host disappears, never blindly restart its worker: held stale jobs can otherwise start fresh into used directories. Preserve old outputs; create a new independent recovery output, hard-link the latest complete trusted checkpoint, retain the same model/schedule/data contract and resume. Native strict DCP retains its legacy campaign_output fingerprint while request.output_dir controls actual export paths. The retired dev13 1025-host jobs remain held.

Latest publication authorization and positive-case request are recorded in reports/authorization.json. Main evaluation for the original-size renderer is being split into three disjoint components on dev13 GPU6 and dev15 GPUs1/2. Read required_components.json before treating that finalizer as complete. Oracle posterior/forecast diagnostic uses source-diagnostic-v20 (508ef20), on dev15 GPU3.
