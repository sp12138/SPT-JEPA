# Active sample-level research — 2026-10-29

Latest user asks to use all24H200s productively for high-performing sample-level methods, while preserving10minute gates and stopping unnecessary work. Main root:/efs/prescottsun/SPT-JEPA/runs/frontier_20261029. Read protocol.json/plan.json, source manifest and actual host processes. Old completion_20261028/robustness_20261029/shortscreen_20261029 remain held and must not restart.

20new trials=10objectives(including original/necessary controls)x2diagnostic seeds. Every trial resumes the same seed-matched original2epoch full state at24982updates and stops exactly25582,600additional updates. Allmodelparameters51,097,134,batch128,sigma=.1,beta=.25,124910stepLRschedule. No cross-sample loss/BN/EMA/target network. Some objectives are explicitly surrogates, not exact KL. Task checks use paired planning and generation/action cases.

4otherGPUslots run selected lower-beta/temporal-forecast seeds1/2 only to2epochs under runs/shortscreen2_20261029; that selection followed actual1epochtask evidence. Both campaigns require a scientific decision before further budget; there is no automatic long continuation.

Source/config snapshots are committed/frozen. Existing Lightning Trainer/checkpoints own all state. train_stage.py adds only a validated stopping callback; fulloptimizer/scheduler/RNG/cursor resume test passed exactly onCPU. Preserve all failed/paused outcomes. Evaluate finite gradients, posterior reconstruction and state interventions, not active_dims alone. P0weakness at10minutes is not universal disproof; inspect the paired continuation baseline.

User prefers30-60minute waits between stable long training checks; near10minute gates/completion use shorter intervals. Publish useful complete numeric outcomes/figures and distinguish adaptation, from-scratch stability and novelty. All files remainEFS except small Git-managed code/reports locally.
