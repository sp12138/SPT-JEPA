# Active24GPUstaged sample-level research

See [current recovery](RECOVERY_FRONTIER_20261029.md). Prior short-screen results are complete; old broad grids remain held. New20adaptation trials stop after600updates, alongside4selected bounded second-stage runs.

---

# Short-budget screening active — 2026-10-29

User corrected the long candidate runs: inspect short-line outcomes before long training. All16runs in robustness_20261029 have STOP markers, full checkpoints and global HOLD_DISPATCH. They stopped successfully around9556-10293steps; status finished/exit0 means graceful pause, not completion of37473updates. Never clear their STOP or unhold that plan.

Current root:/efs/prescottsun/SPT-JEPA/runs/shortscreen_20261029. Only seeds1/2 of four methods continue from full state to12491updates. The LR schedule remains124910steps. Existing Lightning Trainer/checkpoint code remains; train_stage.py provides a validated exact-step stop callback, because the old trainer bypasses stop_after during resume. New outputs hardlink parent full checkpoints without modifying them.

Compare to existing original beta=.25 epoch1checkpoints at runs/stability_20261029/snapshots/trainseed1/step0012491 and trainseed2 analog. Eight candidates plus two originals use the same P0reset identities and compact generation/action diagnostics. At least both seeds must be considered; this is a selected diagnostic screen, not a final replication.

No automatic long continuation. At the first-stage report, select at most one or two directions based on paired P0, action following, generation and cost; defer clearly harmful methods and leave ambiguous ones explicitly inconclusive. Preserve seeds3/4 paused until justified. Publish useful results and key scientific figures.

All code/data/checkpoints/logs remain under /efs/prescottsun/SPT-JEPA; local storage is Git code/small result artifacts only. User prefers30-60minute waits when nothing else is needed.
