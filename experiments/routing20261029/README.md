# Same-size sample-level gradient routing

See [recovery](../../RECOVERY_ROUTING_20261029.md). Three targeted semi-gradient candidates, one renderer-grounding negative control, and the original control use four paired existing training seeds. Forward loss values and parameter tensors match; only gradient routes differ. All20adaptations stop after600updates before budget review. No EMA, target network, cross-sample loss or extra model parameters. This adapts known stop-gradient principles and is not a novelty claim.
