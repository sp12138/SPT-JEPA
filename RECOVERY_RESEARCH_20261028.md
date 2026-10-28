# Major conclusions complete — 2026-10-28T18:57:13.226920+00:00

User scope: finish the major conclusions and necessary direct controls, notify the user, leave other work deferred.
The completed report is experiments/completion20261028/results/MAIN-CONCLUSIONS.md.
Checked delivery: 46 completed endpoints, 30 paired comparisons, eight independently confirmed collapsed checkpoints, three independent I08 pretraining/finetuning seeds and same-weight feedback controls.
Authoritative completion evidence: /efs/prescottsun/SPT-JEPA/runs/completion_20261028/MAIN-CONCLUSIONS-COMPLETE.json
Frozen reporter source: /efs/prescottsun/SPT-JEPA/runs/completion_20261028/main-report-source-v2.json
All new dispatch is held after completion. Do not resume other original grids, capacity/20-epoch extensions, extra ideas or unrelated repairs without a new user instruction.
The original full-budget protocol remains incomplete. Historical long-continuation errors and deferred jobs are preserved; do not relaunch from old handoff statements.

---

## 最新优先级，2026-10-29（北京时间）

用户要求关键 insights 优先：持续坍塌保留证据和可恢复检查点后早停；未启动长网格、额外种子及条件扩展暂缓。优先补候选共同基线、三个 epoch checkpoint 生成/规划、I08 匹配复现。完整原方案仍未完成，暂停项不是成功或失败。实际调度见 EFS 的 early-stops/、scheduling/insight-priority-20261028.json 及 plan.json。数值及图见 experiments/completion20261028/results/EARLY-STOP-TRIAGE.md。

# Active completion campaign,2026-10-28

User authorization is settled: finish outstanding original experiments and validate ten methods; dev2/dev13/dev15 have no campaign time limit. Publish useful additional complete numerical results and key figures. No further scope, time or push approval is needed. No subagents are authorized.

Authoritative root:`/efs/prescottsun/SPT-JEPA`. Code:`code`, branch:`research/ideas-20261028`; campaign:`runs/completion_20261028`. The mutable durable dispatcher plan is `plan.json` with deadline=null. Inspect `workers/*.json`, `exits/*.json`, per-run dispatch/source/config and actual GPU/processes before acting. Detached worker PIDs at last check:dev2 1976783,dev13 64624,dev15 90563. Never launch duplicate workers. `HOLD_DISPATCH` would stop new launches only.

Local SSH proxy briefly reset; all hosts recovered2026-10-28 05:08UTC and active training persisted. Control sockets:`/private/tmp/spt-jepa-dev2-20261028-ctl`, similarlydev13/dev15. SSH aliases remaindev2/dev13/dev15. Interpreter:`/efs/prescottsun/project_jepa/envs/lewm/bin/python`. Do not change the shared interpreter.

Running/queued: lowbetaL2seed1/2; L0seed2 complete; L1MLPBNseed1/2; g0/state128;NativeS0/S1seed1/2; S2beta.0025/.01/.04; originalrenderer full-state continuation; matched research50k arms;geometry and fork/null candidates. Plan inventory is authoritative,not this static list.

Full data:`data/pusht-lance-full-v1`, phase0evaluation view:`.../phase0`; ready=true. Rawactions metadata was completed without changing pixels. 2,336,736rows,18,685episodes,934test; no split overlap. W8 steps/epoch12491;10epoch124910; collapse stage37473,8epoch99928. FullL2/L0 GPU smokes exit0. Original fullL2 grid was activated after finite-loss and LR checks; audit:`full-smoke-activation.json`. All launched snapshots immutable.

Frozen sources: training oldv14; completionv2;researchv3;methodv4;planningv5;completionv6/v7/v8. `completion-source-v8.json` identifies35d069d;19tests pass at`contracts-v8-unittest.log`. Initialpytest runner failed for missingpluggy,so do not call that a model failure. Use unittest. Corrected carry GPU:`idea-I08-native-carry-smoke-retry1` exit0; firstfailedsmoke preserved. Quality postprocess exit0. ExactNativeS0/S1 25k exports:`endpoint-source-S0-25k-seed0/model-step0025000` andS1analog,loaded251/253tensors using standard torchDCP.

New checked endpoint evaluator and all-target-frame formalcollapse audit are being frozen into a new evaluation snapshot. Stage artifacts, nestedprobe errors, exactstep and weightidentity are checked. Endpoint jobs must be registered after their parents; most historical/extra runs still need these registrations. Full-grid stage3 must use the formal noisy posterior audit,then resume noncollapsed full training in independent dirs with same optimizer/schedule state. Do not markall complete.

Completed additional evidence: quality`quality-50k-seed0-full256/results`, FVD278.20946/copy1628.625;sampleFVD286.183/282.164/278.326/289.321 in`quality-50k-seed0-postprocess/results`. Protocol256 clips,3context+13generated,notRGB16FVD2048. I10GoalMSE seeds142/143/14464/78/72%;covariance4/10/8%;Gramian6/12/8%;same training model,paired newevalcases,strongnegativeevidence. Preserve protocols,timings and all outcomes.

FormalgalleryAIreview existsat`reports/completion-qualitative-review`;human_review_complete=false,formal_gate_passed=false. Prepare final reviewable galleries before asking human signoff. AIinspection is not user review.

Keep code,configs,aggregateJSON and keyPNG/SVG in Git; all datasets,logs,weights,rawarrays,checkpoints andvideos stayEFS. Source/config snapshots already launched are immutable. Training/checkpoints remain owned by existing Lightning/Native implementations.

## Latest durable additions,2026-10-28 06:25UTC

- Active plan currently222jobs,including originalfullL2 stage3/conditional10epoch continuations,fullNativeS2grid,fullNativeS0/S1seeds and fullLeWM baselines. Original selected extra seeds,edge-grid extensions and20epoch decisions remain dependent on future scientific selection;do notdeclarecomplete.
- `control-source-v12.json` points2b101a6;source-control-v12 uses immutableNative runtimev11. Detached metadata gate monitor PID2003266 ondev2,recordedcampaign-state-launch.json. It validatesNative fullsmokes,setsnoncollapsefamilymarkers,andrefreshesreports/progress_latest.json. It does nottrainorauto-selectfinalconfigs.
- v11source def169a,29CPUcontracts pass. NativeS1fullGPUbatch64xaccum2smoke exit0 andgatePASS. FullNativeS2smoke active,fullS0smoke queued. Bothnewdensity64stepboundandanchorcolorcandidatespassed200stepGPUtraining;density50kstarteddev15GPU2. Useactualsamplerlabel;64reverseGaussiantransitions+discreteendpointisnot50Euler.
- NewI01vdm_nats includesGaussianprior,discrete8bitreconstructionandfiniteGaussiantransitionKL;negativeELBO/nats,notexactNLL. I04anchor_canonical_colorusesanchorchannelmin/maxonly;spacepermutationcontractpasses. ExistingweightedFM/framewisequantilearmsremainseparatecontrols.
- I10exactcontextcachefix rerunscomplete:6/12/8%,mean8.67vsGoalMSE64/78/72%,mean71.33. OldGramianmomentcacheinvalidatedandpreserved. Statsrecordactualextra_rollouts.
- Formalresumecontrol:`resume-control-v10/PASS.json` confirmsuninterrupted4CPUstepsvs3+1newdirectoryfullstate,weightmaxdifference0,identicaloptimizer,scheduler,batchcursor. ConditionalfullL2continuationsrequireboththismarkerandformalstage3PASS.
- All6new25kbaselineendpointsfinished:S0/S1seed1/2,L1MLPBNseed1/2. L1P0means27.33/16%;NativeS1agentprobe~.75/.80buth7agenterror/copy5.97/4.94. ExactNativeS0/S1seed0endpointsandL0matchedseedsalsoevaluating.
- AdditionalcompleteJSON,per-casenumericalNPZ,andPNG/SVGfigurespreparedin`experiments/completion20261028/results`;reporterscommitted.Source-report-baselines-v13.pypins3445b44. Mainreportsource-control-v12. FigurevisualQAdone.Nohumanformalgallerysignoff.
- Formalnoise-awareprobesfixanisotropicQ;GaussianchannelboundwhitensQ.Markovvarianceconstantisexcludedfromactivevariable-ratecounts.CPU3testsPASS.Noactiverunssource/configswereedited.

## Full-protocol recovery,2026-10-28 07:35UTC

`followup-source-v14.json` identifies9825b59;all39CPUtests pass incontracts-v14.log. FullNativeS0/S1/S2 GPU200step smokes are all exit0 andhavegatesPASS. Current plan has1325entries:222previouswork plusprespecifieddormantfollowups/onefallbacksmoke. Thisisnot1325actualtrainingruns. Mostnewentriesstayblockeduntilapriceisactuallyselected,anedgeextensionisneeded,afixedall-pricescollapsefallbackistriggered,orcapacity/20epochrulesapply.

`full-protocol-registry.json` contains46trialpossibilities,default/rampandfactor4edges;only13initialpricesareactiveforfullcohortnow. `full-protocol-launch.json` pinsPID145185 ondev15,sourcev14;watcherusesonlymetricsandemitsselectionmarkers,itnevertrains. Existingper-hostdispatcherPIDsremainunchanged. Finalprice selection appliesoriginalgeneration/tieplanning/rateorder;closescore seeds3/4,L1SIGRegscreening,formalhumanreviewandfurthermechanisticfallbacksremainrequiredstatisticalreviewitems,notimplicitlycomplete.

Nativefixedprice-rampfallbackhasanew200stepGPUcheck`full-S2-ramp-smoke-v14`queuedondev13GPU0beforeanyrampgridcanlaunch. Onlydormantjobsdependencylistswereupdated.Allrunningconfigs/sourcesremainedunchanged. Sourcev14manifestcoversruntimeandtools;39testsuccesswasrecordedbeforeanSSHcontrolchannelclosed,soexit255ofthatlocalSSHwasnotatestfailure.

MainnewL2seedswere~45k/50kat07:22UTC,finite. Endpointevaluatorsqueuedautomatically. Originalrendererreached25kanditscompleteendpointfinished. Sharedrootandmonitorstatusremainauthoritative.

Additionalconfirmationwindowscodeisbeingfrozenintov15:selectionseed20261028andremovalofallstartswhose16-frameclipswouldoverlapoldfixedcases;CFseed142excludesoldcasetimeintervals;P0confirmation242/243/244. Thisisthesameheldoutepisodesplit,notanindependentdatasetorthreetrainingseeds.

## Confirmation/publication continuation,2026-10-28 07:45UTC

- Evaluation snapshotv15:`confirmation-source-v15.json`47c55cf;runtime dependencyfullv14. Two targeted tests prove deterministiccaseIDs,balancedmotionandtemporalnonoverlap. Sealedbank:`reports/confirmation-case-bank-v1.json`,256windows,2785overlappingoldstartsblocked.
- Additionalconfirmationjobsdependoncompletedcandidate/matchedbaselineendpoints;oldqualitycasesareexplicitlyexcluded,CFusesoldcaseexclusionfile,andP0uses242/243/244. BothI08same-weightsfeedbackcontrolsandI02permuted-Qcontrolareincluded. Originalselectorgenerationcasesunchanged.
- I09fork/null eachhasanew200stepGPUcheckbeforeitslongrun;`idea-action_forks-smoke200`andnullanalog.Noalreadyrunningconfigsorcommandschanged.
- Source959c179 pinsreport_baselines.py;actualfrozenreport:`source-report-baselines-v16.py`. Additionalbaselinepackagecontainsall9L1/S0/S1seed0/1/2qualityendpointsplusall3matchedL0planning/probeendpoints. L0P064.22±9.10%,L125.78±9.10%;stdacrosstrainingseedsaftereachtrainseedaverages3evalseeds. Generationagent/copy:L1.987±.025,S01.080±.042,S13.989±2.595,all25knotfull10epoch.
- Planhas1349entries,mostlyconditionaldormantconfigs. Do notreadthisas1349trainingruns.All24GPUsremainallocatedtoactualactivejobs. Noautomaticretryoffailedruns. OnlyunresolvedhistoricalfailedrunisfirstI08smoke;successfulretrypreserved.
- Publishedprioradditionalpackage4e9fb86,thenprotocol9825b59;latestnewbaseline/casebankartifactsarebeingcommittedandpushedsamebranch.AuthoritativeEFSgitcleanlinessandpublicationrecordmustbecheckedforlatestrevision.

## Exact50k endpoints,2026-10-28 07:50UTC

L2beta.01seed1/seed2andg0seed0allfinished50000,exit0,fullcheckpointsandportableexportsintheirownrunfolders. Evaluationdispatchbegan:g0probes/collapseondev2GPU3;seed1waitsforfixedNative-rampGPUcheckondev13GPU0;seed2waitsfornewcaseconfirmationbaselineondev13GPU1. Bothqueuedendpointjobsarealreadyregistered,do notduplicatelaunch.

Read-onlyliveaudit`reports/active-immutability-audit-latest.json`verified24activejobsacross8distinctfrozensourcesandallconfigsha256,withfreshworkerheartbeats. Baselineconfirmationnew256caseswasat193/256sampledwhenchecked;Nativeexplicit-rampGPUcheckat179/200andfinite. Finalmainmetricsarenotyetavailable;trainingexit0aloneisnotpositivescientificevidence.

## Finished lightweightreplications andnewcandidateevidence,2026-10-28 08:20UTC

AllthreeL2trainingseeds50kandg0nowhavefinishedcompleteendpoints. ExactnewendpointP0:seed068.67,seed112.67,seed258.67%,mean46.67±29.87%acrosstrainingseeds. Meanagent/copy.411±.252,movingblock.258±.022;actionID74.21875%mean<75threshold,soformalworkinggatefails(alsonohumangallerysignoff). seed1isnotempty(noisyposteriorcollapsePASS)butweakagent/actioncontrol. Donotdropthebadseed.

seed0newclipconfirmationfinishedallstages:P0242/243/244=68/66/66%,CF86.71875%,h7agent.1260,block.19325,FVD230.72/copy1345.11. Sameoriginaltestepisodes,temporallydisjointoldclips,notsuccessacrosstrainingseeds. report-confirmationv17andreport-stabilityv20arefrozenscripts;bothnumericpackageandfiguresgenerated.

I08semanticcarryandmatchedpixelcontroltrainingfinished25kfinetunefromsameexact25kS1;completequality:carryagent/copy.18207,block.78110,FVD202.20;matchedpixel.94379/1.12907,FVD939.69.CFstillrunningatlastcheck,same-weights pixel-feedbacktogglequeued.Do notattributegaintofeedbackuntiltoggleandCFarechecked.

MediaCPUexport`gallery-seed0-complete-v18`exit0:32fixed+8best+8worst,allactions/errors/seedsmetadata,andfiveplanningvideosofeachoutcome. Humanreviewpending. media-source-v19=9841996 extendsfutureendpointswithgallery/swap/counterfactualmedia. Allnever-startedevaljobsonlywereupgraded(anddependonnewGPUmediasmokes);completed/runningendpointssnapshotsuntouched. Additivemediajobsfilloldendpoints. Duplicatefuturedormantmediajobsremovedbecausev19alreadygeneratesthem.

NewmediaGPUsmokes`media-v19-swap-smoke1`,`media-v19-counterfactual-smoke1`werequeuedondev2GPU2aftershortI08pixelendpoint(toavoiddelayingbehindlongfullgrid). Theygatefutureendpointlaunches;watchtheseandfixfailuresinnewsource/retrydirs. Currentplan1414entries,mostlydormantconditionalprotocolarms,notactualtrainingcount. Auditlastchecked24activejobs8frozensourcesPASS.

## NewNativeS2numericalfailureaudit

ThethreeexploratoryS2beta.0025/.01/.04trained50kandexit0butcontainNaNweights;endpointprobesproperlyfailed. Firstnonfinite100-steplogsat28400/29100/48600updates;verifiedlastfiniteportable28k/29k/48k. FulloptimizerDCPonlylatesttwoandnonfinite,soportablewarmrestartisnotfullresume. Newretriesarefromscratchathalflr5e-5,withnewcommittedfinite-loss/exportguards. Preservefailedrunsandclassifyasnumericalfailure,notemptycodeorfinishedvalidmodels. FulloriginalS2beta1runwasfiniteat6499andusesuncalibratedinitializer,differentcohort;do notstoporresumethatfromtheseweights.

## I08positivecontrol,capacityfailureandnumericsretries,2026-10-28 09:00UTC

I08carryseed0completeCF83.59375%;matchedpixelCF31.640625%. Same-weights-pixeltogglequalitycomplete,exactcheckpointsha04567348a36686bad49bb3836bc2d98fa0a7747b22fb573ac7e476effc9192cdmatchessemantic;first-stepagent/block6.115/3.984pxidentical. h7semantic24.152/25.269pxvspixel125.200/45.770px;h13agentsemantic137.579vs129.800px,worse,soonlymidhorizonbenefit.SameweightCFstillrunningatlastcheck. source-report-semantic-v23.py=8a8596d;publishedJSONexplicitstagepending. I08pretrainingseeds1/2+matchedfinetune/togglejobsregisteredinfreshfolders.

Capacity128seed0complete:agent/copy.93949,block.41375,actionID39.0625%,P012/14/4%=10mean;all128activeandnoisyposteriorcollapsePASS. Reportstatecapacityvs64includesnats/computecomparabilitycaveats. source-report-capacity-v22.py=c64f689.

Currentcode79f74b0passesall43CPUtests:`contracts-latest-v24.log`. mediaGPUv19swap1/counterfactual1both exit0. fixed/best/worst/planningCPUgallerycomplete. Noformalanthropomorphicclaimofhumanreview.

S2failed50kweightsforensics:`reports/native-S2-numerical-failures.json`,firstbadlogged28400/29100/48600. GuardsinNativeandloadevaluatorcommittedcf915f8,source-numerics-v21. FreshhalfLR5e-5S2seed0runsand200stepsmokeregistered(noimputation,warmresumeorNaNrepair);finite29kweight-onlydiagnosticseparate. FuturesafetrainingNativejobs203only(notlaunched)wereupgradedtoguard-snapshotv21,sameconfig/modelmath,oldrunningimmutable.
`source-numerical-watch-v24.py`79f74b0isfrozenanddetachedoneperhost:dev22042476,dev13150970,dev15163953.ItreadsactualhostworkerPIDs,requirestwoconsecutivenonfinite losses,verifies/proccmdlineandprocessgroupagainsttheexactauthorizedjob,thenSIGTERMsonlythatjobgroupandwritesnumerical-stop.json.Nohealthyjoborotherprojectisaffected.

GivenL2/L0meanP0difference17.56<2*largertrainseedSD59.73,seeds3/4ofbothexploratoryconfigsarequeuedperoriginalrule,notusedtohideearlierfailure. Allnewrunsunlimitedtime. I09fork/null200GPUstepchecksbothexit0,longrunsscheduledaftershortsameweightfeedback/mediawork. I07predictableweightdot50kendpointfinishedbutpoorgeneration(agent1.20copy,movingblock1.36,FVD6179);awaitmatcheddotbaselineforattributablecomparison.

Currentplan1440entries,mostlyprespecifieddormantpricefallback/selectedfollowups,not1440trainingruns. Mostrecentrealstatusall24slotsactive;EFSmonitors/plan/exitsremainauthoritative. ReportsandplotsunderGitmustbesynchronized/pushed;weights/logs/mediaonlyEFS.
