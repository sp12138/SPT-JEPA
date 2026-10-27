# Active continuation — 2026-10-27T20:42:41.460309+00:00

Do not stop at a status answer: finish the agreed campaign, reports, commit and push. No subagents used/authorized. The user explicitly authorized publication and then asked for complete numerical results plus key insightful figures in Git. No repeat permission needed.

Deadline unchanged:2026-10-27 22:24:58UTC / Oct28 06:24:58 Shanghai. Authorized dev2, replacement dev13, dev15 (8H200 each). Authoritative ROOT=/efs/prescottsun/SPT-JEPA; CODE=ROOT/code; C=ROOT/runs/insights12h_20261027; PY=/efs/prescottsun/project_jepa/envs/lewm/bin/python. No new training. All late extra work below is inference on immutable pins.

Publish ONLY https://github.com/sp12138/SPT-JEPA.git, branch insights/12h-20261027. Local origin and earlier push dry-run verified; nothing actually pushed. Never upstreamSPT/main/force/PR. Local clone is still8419475; fetch EFS branch and FF after final commit. User prefers all runtime/data EFS; local small code/presentation only.

MAIN STATUS 28/28 complete (including component dependencies). Pending:

Original renderer primary on dev13GPU6 has required_components.json: actions/P0/CF on dev15GPU1,64x4 sampled generations on dev15GPU2. Wait for BOTH before completion. Read actual processes/worker JSONs, not stale handoff.

MAJOR NEW POSITIVE:50k tiny model WORKS in the numerical tests. This changes the report, whose staged prose is still an obsolete draft centered on25k failures.
- Model L2-standard-calibrated-b001-50k-seed0:21,580,398params,64dim state, beta.01,std-init.1,sigma.1, from scratch(noL0warm/noBN/noSIGReg). Fresh50k cosine budget, same model as25k short control; config differences only max_steps/output/max_seconds/export_initial.
- Noisy agent/block R2=.867517/.971239, FP32mean .870612/.974067.
- Final256 mean windows: h7agent/copy=.233920, h7moving-block/copy=.233695 (128moving); h13agent/copy=.2131. h7absolute agent31.03px,block10.95px. Positions use512x512 simulator coordinates; images64x64.
- Four latent samples per case: h7agent=.265263,movingblock=.260950. Both mean and sampled numerical generation conditions are promising.
- Heldout P0 seeds42/43/44=74/76/60%, mean70%, vs matchedL0 seed0 mean74.7%,seed1 63.3. Same cases. Do not claim statistical equivalence or compute-matched performance (L2~4.6GPUh vs L0~.6GPUh).
- Counterfactual64x4:overall actionID=.8828125,agent=.859375,block=.6875,chance.25. Real-frame pose reference=.875; this is an evaluator reference, NOT a mathematical ceiling.
- Visual inspection of first16 fixed moving cases shows agent/block remain recognizable through13steps with noise and some geometric errors. No cherry-picking. Original gallery on EFS and localpreviews/standard-beta001-50k-first16.png.
- This is ONE training seed. Original3-training-seed gate not established; single-seed numerical generation conditions are satisfied. Do not keep saying allASJEPA fails or agent is always unlearned.

NEW SAME-TRAJECTORY AUDIT (all5jobs complete, no training):
long_budget_diagnostics_registry.json lists decode-decomposition-long50k-dev2, budget24k-main-dev2, budget24k-planning-seed42/43/44-dev2.
The earlier24k root export had been pruned; recovered exact immutable weights from parent/eval-posev2/step0024000/checkpoint, pinned+SHA in budget24k-main-dev2/checkpoint_manifest.json. This is SAME continuous50k run, not a new training trajectory.
- 24k noisyR2 agent/block=.652772/.960987; mean generation256 h7agent/copy=.859908,movingblock=.322026.
- 24k P0=32/22/18%,mean24%. Thus SAME trajectory24k->50k improves agent .860->.234 and P024->70. Independent25k-short has different cosine budget; do not attribute all short-vs-long difference purely to count.
- New50k posterior decomposition128cases:posterior agent h1/7/13=13.96/36.16/28.79px; forecast18.32/36.00/40.18; posteriorblock4.99/5.50/5.22 vs forecast5.13/10.69/18.88. Compare old25k posterioragent116.6/127.9/133.8. Encoder/decoder route improves strongly with budget; prior still adds accumulatingblock error. Truefuture posterior is diagnostic, not deployable prediction.
- New diagnostics reporter changes UNCOMMITTED: collect all decomposition-*roots, add50k column, matched budget evidence4-panel plot+CSV, label first16 50k gallery. verify/export changes add new registry and budgetcase/weight checks. COMMIT THESE, freeze CPU source-report-v31, run on actual data, QA new plots. Source-v30ff1e4d3 was previous frozen CPU subset.

All earlier auxiliary results complete:
- late-probes-standard-dev2:9models36mean/noisy BF16/FP32 probes; native3models8probes; no errors. NativeS0/S1 at24k have useful pooled R2~.83/.88 and .81/.81; S2 beta4 at12k noise chance.
- NativeS0/S1 quality256 cases,50Euler,pose-v2; firststepagent12.48/9.24px vs copy44.91, later drift.
- Native true-past audits256 permodel (native_context_registry):h1percase error maxdiff0 for bothobjects. S1h7 teacherforcedagent/block16.69/6.57 vs free145.39/58.12. Same sampler/actions/anchor/noise, only past frames privileged. Source-v24=711463f.
- Native clip-history256 S1(source-v26f703157):h1diff0; clipping makes h7agent273.18,block90.73,worse. Keep as inference ablation, canonical model unchanged.
- Color geometry4models128cases(source-v27=8c38090):old palette+ownpose scores exactly0difference. Same reversed donor pairing for separate anchor/state interventions (protocol doc corrected; not independent permutations).
  Anchor seed0 ownblock errors:own33.79;swapanchor33.59;swapstate to own113.70/to donor37.46. Seed1 own37.34;swapanchor40.14;swapstate own114.12/donor38.70. Demonstrates background-from-anchor, blockposition-from-state with imperfect geometry, not fullagent disentanglement.
- Color beta.25/std1 two seeds:palette transfer anchor .9447/.9444,state .00093/.00050; noanchor state .9463/.9451. Noisy blockR2 .902/.889 vs noanchor .845/.519, paletteR2 .194/.139 vs .586/.510. Agentmixed. Paired movingblock generation delta anchor-noanchor -.127 CI[-.189,-.070],-.393 CI[-.536,-.267]; agentCIsinclude0.
- Dot15k beta.01:flow retainsRGB R2~.8/.83/.83, MSE/L1BN~0, positionnuisance~0 all. Flowblockprobe .834>MSE .677, yet movingblock generation .779>MSE .472. Lowprice finding, does not settle beta>=1 theoretical conditions.
- High gain BF16variance falsepositive:meanstd.15-.2 vs FP32~1e-4, noisyFP32 task chance. FullFP32 beta.25 noisy.206/.207 but h7agent1.114,movingblock1.227,P0mean4.7. Higherprecision not sufficient.
- Default correctedbeta4 full25k still empty; beta1 continuous25k noisy .445/.833, movingblock .653,P0mean6.7. Early8k collapse not permanent statement.
- Warm encoder+freeze4k beta4 preserves noisy .416/.748,movingblock .814, weakplanning; warmalone loses state. Both use campaignL0pretraining, not from scratch. Include row in final main plot (source-v30 did).
- CorrectedL1noBN25k R2 .339/.656, h7moving .536,P0mean11.3; legacyL1 .094/.099,P02 separate sourcecohort. BN64means16.0/24.7; MLPBNseed0 mean33.3. L0matchedmeans74.7/63.3;legacy82.7 separate.

REPORT STAGING and sources:
CODE/experiments/insights12h/results/ is UNTRACKED draft (~2MB):INSIGHTS.md obsolete lead/pending2budget, README.md draft, diagnostics.json(oldno50kdecomp/budget),late-probes.csv,figures17files from ready_1925. Includes selected25kfavorablePNG; label as selected10of256, not formal success. Need replace/add new50k qualitative evidence and newbudget/posterior figures.
C/review/ready_1925 source-report-v29=8a0a798:alloldauxdone9plots; geometrylegendfixedandinspected. Main26/28partial. v30=ff1e4d3 adds explicitwarm/freeze4k beta4 row, final main plot stillneedsQA. NewCPU sourcev31 will include newbudget/decomp logic.
Environment-versions.json and evaluation-controls.json staged:Python3.12.3,torch2.9.1cu130,transformers5.8.0,lightning2.6.6; upstream21446f1 clean verified. H200allhosts8each; dev2/dev13driver580.167.08,dev15driver595.91.07. No credentials/environmentvars copied.
Data1000pixelchecks pass,474498frames,17751train/934test,nooverlap,W8windows326593. Pose-v2 real10kfloors standard2.37/1.48,color3.03/1.80,dot3.25/1.78. V1dot108pxinvalid. Nine CPUcontracts passed throughunittest13.312s(source14); initialpytest didnotstart missingpluggy, documented. Do not rerun unchangedcontracts. CPU5+15vs20 exactweights0,samples/scheduler match. All903 inheritedruntimefiles unchanged; GPUbitwisecontinuation notclaimed.
All positionerrors512x512scene units; rendering64x64. Originalpaperbudget/phases not fully reproduced; formal3seed stats not done. LPIPS/SSIM/FVDnot measured.

NEXT:
1.Commit uncommitted3reporters plus updatedhandoff; create source-report-v31 subset of __init__,summarize_final,summarize_diagnostics,verify_campaign,export_campaign from that commit. Run bothsummaries+partial verification inNEW review dir, QA budgetplot/3-columnposterior/50kqualgallery. Update stagingdiagnostics/figures/budgetCSV.
2.Wait originalrendererfinalizer+bothcomponents; inspect actualgalleries and fullmetrics. Read-only observer localexec session73053,remotePID1968707 prints45s; currentonlymain27/28expected. It exits READY only componentscomplete.
3.Newfinalreviewdir: committedv31 main summary,full aggregate exporter,verify_campaign --require-complete; copy staticdiagnostics from verifiedv31dir if inputsunchanged. FullaggregateJSON~6MB allowedsmallnumericartifact; rawarrays/weights/logs/videos remainEFS. Mainnumbers+intervals+sampledstatefields and allhistoricaltier/invalidflags included.
4.Rewrite INSIGHTS.md around50kpositive+budgettrajectory,anchorroles,Nativefeedback. Retain scoped25kfailurediagnostic instead of claiming persistentASJEPAfailure. Addbigcontroloutcome; preserve one-training-seed limit.
5.Close only owned workers/evaluator afternecessarywork, actualprocess/GPUcheck. Can archive plan thenholdremainingdispatches to prevent restartingoldNative outputs. No newcompute beyondoriginaldeadline.
6.Commit fullnumeric+keyPNG/SVG+metadata/docs inEFS code,updateHANDOFF/README. FetchEFS->localFF (localstill8419475),push verified SPT-JEPA branch. No approvals needed. Final linkreport/branch and strongestpositiveinsights.
