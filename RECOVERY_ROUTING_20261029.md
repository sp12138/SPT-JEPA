# Bounded gradient-routing screen

Authoritative root: /efs/prescottsun/SPT-JEPA/runs/routing_20261029. Read protocol.json, plan.json and source-v1.json, then actual worker/GPU states. Twenty trials are five forward-identical settings (including one negative control) x four existing training seeds. Each resumes its seed-matched2epochfullstate at24982 and stops25582. No automatic long continuation. Four separately evidence-selected shortscreen2 runs retain their own GPU slots and stop24982.

All code/data/logs/checkpoints/recovery live in EFS. Never restart old held grids or modify running snapshots. Existing Lightning trainer/checkpoint implementation remains unchanged. Restore alloptimizer/scheduler/RNG/cursor state. State tensors include fixed buffers beyond51,097,134modelparameters. Check posterior reconstruction and state interventions, not activity alone. Stop-gradient routing remains sample-level but changes the full-gradient algorithm. Seeds1-4 are existing mature models; adaptation is not a from-scratch stability result.

First-screen full numerical results and figures: experiments/frontier20261029/results. All120endpoints completed0 and no method consistently improved P0 in both diagnostic seeds. That grid is held and its sources remain immutable.
