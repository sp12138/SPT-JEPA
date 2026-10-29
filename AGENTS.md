# SPT-JEPA current execution policy

Reviewed six forecast-only short gates: full-state/gradient/posterior checks pass and fresh initial tensors match original controls. `forecast_firstepoch_20261030` resumes only these six runs to update 12,491, then stops for P0/health/generation/CF review; no automatic second epoch. It uses dev2 GPUs 2/3 and dev13 GPUs 0–3. Preserve the parent's two live third-epoch forecast runs and the other 16 replication slots. New-case third-epoch results are published; two mature training seeds do not establish broad stability or academic novelty.

Reviewednewseed5-8firstepoch:96jobscomplete0,all16fullstateauditspass;planningstillimmature,combinedagentgenerationbetterall4. promising_newseed_epoch2_20261030advancesall12original/lower/combinedrunsonlyto24982,withsamecasesandallseedspreserved. Other4seed34and8forecastconfirmationslotsremainseparate. Noautomatic3epochforthesenewseeds.

Current8freedslots:forecast_confirm_20261030checksnewP0seeds2042/2043/2044,2forecast-only3epochrunsand6forecast-only600gatesbasedoncompleted2epochtaskgain90/90. Parentpromising112jobscompleteandheld. Otheractive12newseedfirstepochand4seed34second-epochslotsremainseparate.NoextraI03longrunsorfailedfrontiercontinuations.

Complete3epochseed1/2results:temporalforecast94/94primary,93.33/91.33additionalvsoriginal88/74.67andretuned86/84. Component2epochforecast-render90/90matchescombined88/90withinuncertainty;testthesimplerforecastpathnext. Allpromising112endpointsexit0andall8budgetcheckpointsfinite. Otherseedreplicationsremainactive;donotpromote2seedgainsasfinalbroadstabilityornovelty.

Additionalreview:existingseed3/4firstepoch24endpointscomplete0,combined15/18P0vsretuned7/5andallhealthpass. promising_seed34_epoch2_20261029advancesonlythose4runsto24982withsavedoriginalreferences;noautomatic3epoch. promising_epoch1newseed5-8and8parentselected/componentjobscontinue.

600updatefollowupreviewcomplete:all16gateauditspass,all3methodinitialtensorsmatchpernewseed. promising_epoch1_20261029 advancesonlythese16runsto12491andstopsfortaskevaluation. The8selected/componentjobsinpromising_20261029continueunchanged. Noautomatic2epochcontinuationfornewseeds;nooldSTOPclearing.

New strongest evidence: complete52selectedstage/endpointevaluations show temporal-forecastseed1/2twoepochP0=88/90vsoriginal73/43;fresh85.33/87.33vs73.33/36.67.Lowerbeta=.0625isstrongretuningcontrol.Newpromising_20261029protocoluses24boundedjobs:4selectedthird-epoch,4existingseed3/4first600,4monocomponentsecond-epoch,12newseed5-8first600. Thisisspecificfollowupafteractualtaskevidence,noblanketoldgridresume. PreserveshortgatesandoldSTOPmarkers;matchedmodelsize/sample-levelloss. Novelty/finalstabilitystillunproven.

Current evidence: anchorconfirmation completed120exits0; the apparent drop25lead did not survive freshcases/numericalrepeat. Do not continue those candidates. Finish only the already evidence-selected shortscreen2twoepochruns and stage2evalmatchedtaskchecks before the next researchdecision. Earlier taskauthorization remains, but do not launch blanket long grids. No newalgorithm/stable89%claim is supported yet.

Current confirmation: anchor_20261029 completed120exits0; drop25primaryP0effects+4/+3/+5/0pp warrant confirmation, not a performanceclaim. anchorconfirm_20261029 uses12same-state600updaterepeats+8fresh-evaluation/probeGPUslots, with4unchangedshortscreen2slots. No automatic longertraining; context-mask is a secondarycontrast. Preserve source799a66c and allolderheldplans.

Current stage: anchor_20261029 schedules20matched600updateadaptations (original,anchor dropout0.1/0.25/0.5,observed-context-mask)x4existingseeds, preserving original full gradients/model size. Three prior short screens completedall120endpoints0 each and remainheld. Fourselectedshortscreen2runs remainunchanged. Past-anchor shortcut is a hypothesis; judge actual task evidence before more budget.

Latest current stage: frontier and routing short screens completed120exits0 each. Preserve their held grids. New actionrank_20261029 runs20same-size600updateadaptations (original+4within-video action-ranking settings)x4existingtrainingseeds, retaining original flow/rate gradients and no extra head. Four shortscreen2GPUjobs remain unchanged. Review tasks/health before any further budget. Avoid novelty/stability claims from adaptation.

Current evidence-driven stage: frontier_20261029 completed all120endpoints with no consistent two-seed planning winner and remains held. New routing_20261029 uses5forward-identical gradient settings (original,3targeted candidates,1grounding negative control)x4existing training seeds. All20adaptations stop after600updates before review. Preserve the separate4selected shortscreen2 jobs. No blanket continuation or from-scratch stability claim.

Latest user instruction: use all24H200s on dev2/dev13/dev15 productively for strong sample-level algorithms. Preserve short gates. Current frontier protocol runs/frontier_20261029/protocol.json schedules20same-size600update adaptations plus4evidence-selected bounded stage2runs; no automatic long candidate grid. Publish useful new conclusions promptly.

Latest clarification: future method trials first save and inspect an approximately600update(about10minute) health stage, including numerical/gradient/state-use checks. A finite200step smoke is not a functional noncollapse result; do not skip directly to long grids. Early not-yet-learned behavior is inconclusive unless persistent failure is independently verified.

Latest user correction,2026-10-29: inspect short-budget method results before long experiments. The16run robustness matrix is gracefully paused with recoverable state. Only seeds1/2 of four methods may finish the first epoch for matched short screening; do not resume the long grid automatically. Current scope: runs/shortscreen_20261029/protocol.json.

User instruction,2026-10-29, renewed reproduction and research authorization:
- Wait efficiently: when no independent work remains during long training, poll around every30-60minutes and sleep between checks; shorten only near completion or an active failure. This is the latest explicit user preference.
- First verify full-data AS-JEPA L2 beta=.25 at three epochs across independent training seeds. If stable, perform valuable remaining Lingjing experiments; if unstable, investigate and develop a more stable/better method at comparable model size. dev2/dev13/dev15 remain authorized. Stop unneeded experiments promptly. Commit and push useful new conclusions with numerical results and key figures.
- This supersedes the earlier major-conclusions-only hold for this new campaign. Old snapshots and the old held plan stay unchanged. Current scope/protocol: runs/stability_20261029/protocol.json.

User instruction, 2026-10-29, major conclusions only:
- Finish the major conclusions and necessary direct controls, then notify the user. Leave other work aside. Do not resume deferred original grids, capacity extensions, additional ideas or unrelated fixes without a new user instruction. The active allowed scope is recorded in runs/completion_20261028/scheduling/main-conclusions-only-20261029.json.


User instructions, 2026-10-29 (Beijing):
- Prioritize key insights. Preserve and early-stop persistent collapse. Defer exhaustive grids and low-value extra seeds; complete decisive controls, existing checkpoint quality/planning and promising I08 replications first. Deferred original experiments remain unfinished.

User instructions, 2026-10-28:
- Continue unfinished original experiments and validate the ten candidate methods. dev2/dev13/dev15 have no campaign wall-clock limit; the previous12-hour deadline is lifted for new work. Publish useful additional numerical results and insightful figures. Preserve all historical run snapshots.

User instructions, 2026-10-27:
- Authoritative code, environments, data, logs, checkpoints, reports and recovery instructions live under /efs/prescottsun/SPT-JEPA. This overrides inherited SPT and parent workspace locations.
- Authorized compute: dev2, dev13 and dev15, 8 H200 each (dev15 added by the user at 2026-10-27 12:04 UTC). Other compute workloads on these hosts may be cleared per user instruction; preserve system services.
- This campaign is insight-first within 12 hours. Use single seeds, common held-out cases, early collapse gates and focused ablations; do not require all three-seed grids. Record every departure from the original spec and distinguish exploratory evidence from final benchmark claims.
- Preserve immutable committed source/config snapshots for running experiments and periodic resumable checkpoints. Never modify other projects or shared inputs.
- Local storage is limited to small Git-managed code and presentation artifacts.

## Inherited baseline constraints

# SPT project entry

SPT starts from the frozen RGB16 pretraining implementation in
/efs/stormyxie/reports/rgb16_scale_budget_20261010. Read README.md and HANDOFF.md
before operating the project. The guide in runtime/AGENTS.md governs framework code.
All project runtime, inference and FVD source is bundled in this repository.
Upstream paths in provenance/HANDOFF are historical evidence, not code dependencies.
Use explicit asset paths and the repository-owned preparation/worker entry points.

Code is managed at /threed-code/prescottsun/SPT. Git metadata is separately stored
at /efs/prescottsun/SPT/git/SPT.git. Put manifests, outputs, logs, caches and
checkpoints under /efs/prescottsun/SPT. Shared upstream data and the supplied
Python environment are read-only inputs; never change another user's experiment.

Use the existing Native model/task/training/checkpoint implementation. Do not
reimplement the Trainer or checkpoint manager. Freeze a committed revision into
each EFS run before launching. Running source/config snapshots are immutable.
Check actual GPUs/processes before scheduling work and use independent output
directories. Never stop unrelated jobs or restart historical jobs from handoff text.

The initial RGB16 training was requested by the user. This does not authorize
capacity ablations or the upstream five-branch tuning/FVD campaign. Report training
loss, throughput and sample/FVD quality separately. An early smoke result cannot
establish generation quality. Keep datasets, weights, logs and media outside Git.

Compute allocation (user instruction, 2026-10-13): use only dev3 and dev8. Do not
use dev0. Keep the current main training on dev3; prefer dev8 for demos and FVD.
The user explicitly authorized FVD on dev8 using the original xie protocol.
