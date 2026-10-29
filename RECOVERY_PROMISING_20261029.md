# Latest: forecast-only six-seed first-epoch confirmation

The authoritative root is `/efs/prescottsun/SPT-JEPA/runs/forecast_firstepoch_20261030`; read its committed plan/config hashes and actual workers. Six short gates passed full-state audit. Resume seeds 3/4 on dev2 GPUs 2/3 and fresh seeds 5–8 on dev13 GPUs 0–3 only to update 12,491, followed by twelve P0 and eighteen health/generation/counterfactual endpoints (36 jobs total). No automatic second epoch.

Frozen training/dispatcher source remains `runs/promising_20261029/source-v1`, commit d18c3dce98082f5a773ab1f2e52fb2744c372ec3. The full Lightning optimizer/scheduler/RNG/data cursor resumes from the completed short gates. Do not rerun preparation or alter any parent STOP markers. Parent `forecast_confirm_20261030` still owns the two forecast-only third-epoch runs on dev2 GPUs 0/1. Seed 3/4 second-epoch and fresh seed 5–8 second-epoch campaigns retain their separate 16 slots.

Published new-case confirmation: `experiments/forecastconfirm20261030/newcase-results/DECISION.md`. Combination P0 is 91.33%/94.00% on new paired queries; this is still two mature training seeds from the same held-out episode pool, not OOD evidence. Forecast-only two-epoch component P0 is 90%/90%; its simpler mechanism remains under confirmation.

---

# Current evidence-supported replication

Root:/efs/prescottsun/SPT-JEPA/runs/promising_20261029. Read protocol/plan/frozenmanifest then actual hostGPUprocesses.24independentjobs:4selectedthird-epochruns,4existingseed3/4first600resumes,4monocomponentsecond-epochcontrols,12newseed5-8original/lower/combinedfirst600fromscratch. Everytrialhasahardstop;noautomaticfullgrid. PreservedoldSTOP/heldplansmustnotberesumed. ExistingLightningTrainer/checkpointcodeownstate;immutablecommittedsources/configs.

Allmodelparameters51,097,134,state64,batch128,allfulldata,LRhorizon124910. Intentionalbeta/rho/forecastchangesexplicit. NoBN/SIGReg/crosssamplenegatives/EMA/addedparametersinL2loss. Forecastingiscausalbutaddscompute. Priorobjective/routing/actionranking/anchorgridscompletedanddidnotconfirmstablesuperiority;donotpromotetheirselectedwins.

Twoepochleadcomplete:temporalforecast88/90primaryand85.33/87.33freshvsoriginal73/43and73.33/36.67. Lowerbeta87/85and81.33/76. Thisistwo-seedsamebudgetevidence,notstable89%orthreeepochfinalclaim. Results:experiments/shortscreen2_20261029/results. Userauthorizationpermitsstagedconfirmationandproductive24GPUuse;do_notaskagain. AllauthoritativestateunderEFS;smallGitartifactsonlylocal.

Reviewed600updategatescomplete:all16finitecheckpoint/optimizer/schedulerandnonzerogradient/stateusechecks;newseed5-8initialtensorsmatchacrossall3methods. Newboundedroot runs/promising_epoch1_20261029 advancesonlythese16runsto12491updates,thenplanning/health/quality/CF. Originalpromisingplan8selected/componentlongjobscontinueunmodified. SamehostGPUslotsavoidconflicts;old16jobsstaycompleteatshortcaps.NofirstepochP0winnerornewseedstabilityclaimyet.

Existingseed3/4firstepochcomplete24selectedtasks0;combinedP015/18vslower7/5,noncollapsed/stateuseandgeneration/actionevidence. Newroot runs/promising_seed34_epoch2_20261029nowowns4boundedsecond-epochcontinuationsplusmatchedoriginalreferencechecks. Newseed5-8firstepochandthe8parentselected/componentjobscontinue. Do_notclearoldSTOPorresumeoldparentplans.

2026-10-30newconfirmedendpoints:parentpromising112jobscomplete0;combination3epoch94/94primary,93.33/91.33additional;forecast-only2epoch90/90matcheswithinuncertainty.Fullnumbers/figuresinexperiments/promising20261029/results. Newruns/forecast_confirm_20261030ownsfreed8slots(dev2/dev13GPU0-3):18newcaseP0checks,2forecast-only3epochcontinuations,6forecast-only600gates. Newseed5-8firstepochandseed3/4second-epochcampaignsretainotherslots. Allparentsourcesimmutable.

Fournewseed5-8firstepochcomplete96endpoints0and16fullstateauditspass;combinedagentgenerationimprovesall4butplanningremainsimmature. runs/promising_newseed_epoch2_20261030nowowns12newseedsecond-epochcontinuations,allstop24982withallseedspreserved.Noautomaticthird-epoch. The4seed34second-epochand8forecastconfirmationjobsretainotherslots. Firstepochparent96jobsheld/closedaftercompletion.
