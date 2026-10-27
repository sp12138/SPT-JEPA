# Campaign continuation — updated 2026-10-27T18:23:06.517424+00:00

Original 12-hour deadline remains 2026-10-27 22:24:58 UTC (Oct28 06:24:58 Shanghai). Authorized hosts are dev2, replacement dev13 and dev15. Authoritative root is /efs/prescottsun/SPT-JEPA. Do not touch other projects or historical SPT jobs. New training is closed. No subagents have been used or authorized. User wants insight-first results and explicitly authorized commit & push; latest user additionally requires full aggregate numerical results and key scientific figures in Git.

Publication destination is https://github.com/sp12138/SPT-JEPA.git, branch insights/12h-20261027. Local origin was verified and push dry-run previously succeeded; nothing has actually been published. Never push to upstream sp12138/SPT. Do not ask for publication approval again.

Current EFS HEAD is 3180850 (export campaign reporter). Source diagnostic-v24 is immutable 711463f3fe9913c73235a637d6624d00b0a43ba6. source_manifest_diagnostic_v24.json pins full archive, Python and PYTHONPATH. Source-report-v22 is a reporting-only subset at 80fa218; do not edit either snapshot.

Main finalizers: 21/28 finished at this check; pending entries below. Parallel original-renderer components on dev15 GPUs1/2 must also finish; read required_components.json. All completed inner probe statuses were scanned: no errors.
- ('L2-standard-calibrated-b001-50k-seed0', 'not_started', 0, None)
- ('L2-standard-b1-seed1-continuous', 'running', 3, 7)
- ('L2-standard-original-renderer-b001-budget', 'not_started', 0, None)
- ('L2-standard-fullfp32-b025-seed0', 'not_started', 0, None)
- ('L1-color-bn-seed0', 'running', 1, 3)
- ('L1-standard-cursorfix-seed0', 'not_started', 0, None)
- ('L2-standard-b4-cursorfix-seed0', 'not_started', 0, None)

New completed supplementary endpoints:
- late-probes-standard-dev2: 9 pinned final25k models,36 BF16/FP32 mean/noisy probes, no failures.
- late-probes-native-dev2: Native S0/S1 portable24k and S2 portable12k,8 probes, no failures. Native S0/S1 pooled-position R2 .832/.878 and .813/.808, so these encoders are useful despite failed long rollouts. S2 noisy FP32 remains chance.
- quality-late-S0-native-seed0, quality-late-S1-native-seed0, quality-late-S2-native-b4-resume2-dev2:256 matched windows,50 Euler,pose-v2, all complete. Their case IDs match primary L2 exactly.
- Native S0/S1 h1 agent errors/copy=.278/.206 (12.5/9.2px), but h7 agent/copy=1.105/1.096; moving-block/copy=1.632/1.191. Single-step prediction works, pixel autoregression drifts.
- native-context-S1-smoke8 passed with h1 agent AND block errors exactly equal to cached baseline (maxdiff0). It reuses Native rollout_frames unchanged; only next_context input is replaced by true past frames, with same actions/anchor/noise. The8 cases are a contract smoke, not final evidence. They are first8 of the256 case ordering.
- Full teacher-forcing audits native-context-S0-native-seed0-n256 and native-context-S1-native-seed0-n256 are registered on dev2 GPUs0/1, source-v24. Read native_context_registry.json and actual statuses; expected finish around18:30UTC. These are privileged-past-frame diagnostics, not deployable rollouts. They need inclusion in the final report/figures. Do not confuse with L2 true-future posterior reconstruction diagnostic.

Newest main results:
- L2 beta.01/std.1 seed0 noisy R2 agent/block=.472/.935; seed1=.458/.890. h7 moving-block/copy=.486/.495, h7 agent/copy=1.010/1.003. P0 seed0=18/10/10, seed1=12/6/4. Positive block dynamics replicated; agent/planning still weak.
- L1 BN64 seeds0/1 agent/block R2=.543/.886 and .676/.854; h7 moving-block/copy=.418/.434; P0=18/14/16 and30/22/22.
- L1 MLP-BN seed0 final: agent/block R2=.588/.897, h7 moving-block/copy=.404, agent/copy=.972; P0=42/32/26 (mean33.3%). This is a labeled fallback, not L2.
- Matched corrected-source L0 seed0 P0=76/78/70 (74.7%); seed1=70/62/58 (63.3%). Legacy source L0 seed0=86/86/76 (82.7%) is a separate sampler cohort.
- beta1 legacy recovery at25k regained noisy R2=.228/.451; P0=4/6/6 and motion~copy. Hence8k collapse does not imply permanent irrecoverability. Corrected uninterrupted seed1 final still pending at last check.
- Color beta.25/std1 at25k, both seeds: anchor-swap palette transfer .9447/.9444, state-swap .00093/.00050. No-anchor state swap .9463/.9451. Noisy palette R2 .194/.139 vs no-anchor .586/.510. Block R2 .902/.889 vs .845/.519; agent .432/.445 vs .453/.328 (mixed direction).
- Paired color moving-block generation delta (anchor minus no-anchor error/copy) seed0=-.127 CI[-.189,-.070], seed1=-.393 CI[-.536,-.267]. Agent differences include0. Conditional trained-weight, episode-cluster CIs, not training-seed population intervals.
- Random-dot low beta .01 at15k: flow keeps dot RGB R2~.8 while MSE and L1 BN approximately0; neither encodes dot location. Flow dynamics probes stronger but moving-block generation worse (.779 vs MSE .472). This does not support selective removal by flow in this tested low-price regime; it does not settle theoretical beta>=1 claims.
- Oracle posterior/forecast decomposition128 cases: L2 true-future posterior agent~117/128/134px at h1/7/13, almost forecast error; posterior block~21/19/19 vs forecast21/30/36. Agent failure already in encoder/decoder route; cannot assign encoder vs decoder individually. L1 BN similar; beta4 empty also under oracle.

Reports prepared:
- review/status_1748:19 completed main finalizers, color pairs2; later counts are higher. Core summary.json/CSV/RESULTS.md and final-comparison, color-mechanism figures.
- review/diagnostics_1754:12 supplementary probe models,3 Native endpoints,2 paired color generation comparisons,3 posterior decompositions, pending0. Source at80fa218, so it predates h1 Native figure and teacher-forcing collection. Three plots visually inspected: posterior-forecast, dot-information, precision-diagnostics.
- Current as_jepa_tools/summarize_diagnostics.py at711463f additionally includes all Native horizons and native-rollout-drift plot, but still must be extended to collect the full teacher-forcing audits once finished.
- review/export_1819: committed3180850 single-file exporter;130 task entries and937 aggregate observations at collection. campaign-configurations.json579KB, all-aggregate-measurements.csv309KB, all-aggregate-measurements.json~6MB. Full historical observations are explicitly tagged; invalid dot pose-v1 scores flagged. Raw details/arrays/logs/media stay EFS. Re-export after completion. Could improve source fallback to execution.json if no actual_source/source_manifest.
- User explicitly wants these complete aggregate numerical tables plus key scientific figures committed. Keep main final tables separate from tagged historical observations in prose. Do not call130 tasks130 training runs.

Local review clone was last synchronized to8419475; fetch EFS branch and fast-forward when needed. Local artifacts/insights12h contains small inspected PNGs: color-mechanism-two-seeds, posterior-forecast, dot-information, precision-diagnostics, previews/positive-selected-beta001-25k, standard-beta001-first16, posterior-forecast-beta001, S0/S1-native-seed0-24k. artifacts/ is ignored; force-add only chosen small final presentation files. All canonical media still EFS.

Remaining work:
1. Check fresh processes and every evaluation return code; finish7 main finalizers + original renderer components, plus Native teacher-forcing diagnostics.
2. Extend final diagnostics reporter/figure with Native teacher forcing, inspect galleries; keep h1 same-noise control and privileged-data scope explicit.
3. Audit final numeric counts, common case identities, pose-v2, inner probe errors and pinned weights/source. No need to rerun already-passing9 CPU contracts without framework changes; all903 inherited runtime files remain unchanged.
4. Produce concise Chinese key-insight report, full aggregate results/source/config index and small key figures. The Native feedback diagnostic, anchor two-seed mechanism, low-beta replicated block dynamics, posterior-vs-prior bottleneck, dot counterexample and precision false positives are the main candidates. Await50k/original-size/full-FP32 before final interpretation.
5. Stop only owned workers/evaluator once necessary jobs finish (or original deadline), inspect all GPUs/processes. No deadline extension.
6. Commit result bundle in authoritative EFS Git, sync local via SSH Git fetch, push insights/12h-20261027 to verified SPT-JEPA origin. User publication permission is explicit. Final response should link report and branch, with key outcomes and limits.
