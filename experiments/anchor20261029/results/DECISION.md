# Anchor information screen: a positive planning lead requiring confirmation

All120endpoints exit0; all20checkpoints/optimizers/schedulers finite, exact25582updates and portable/full tensors equal. All20pass the declared coarse posterior-collapse rule. Same51,097,134parameters and original full flow/rate gradients,600extra updates from the seed-matched2epochfullstates.

| Setting | Seed1 P0 | Seed2 P0 | Seed3 P0 | Seed4 P0 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original |76|48|30|83|59.25|
| Anchor dropout0.1 |77|51|31|83|60.50|
| Anchor dropout0.25 |80|51|35|83|62.25|
| Anchor dropout0.5 |76|51|31|81|59.75|
| Observed-context mask |77|47|34|85|60.75|

Drop25gives+4/+3/+5/0percentagepoints, mean+3.0pp, and is the strongest planning lead in these short screens. This is a point-estimate pattern on selected existing models and100common episodes per trainingseed. It is not stable89%performance, an independent from-scratch replication, or a proven causal mechanism. Conditional per-seed bootstrap intervals and complete episode outcomes remain in theJSON; selection among earlier settings is exploratory.

The reconstruction/action results are mixed. Drop25agent/copy worsens in allfour point estimates; moving-block/copy improvesseed2/3/4but notseed1. Agent counterfactual identification improvesseed2/3, is flatseed1and worsensseed4; block identification also has mixed directions. Therefore do not call this a uniformly better generative worldmodel.

The primary benefit becomes smaller after both evaluation seeds: first-eval-only gains+8/+6/+4/+4became+4/+3/+5/0aftereval43. The context-mask's seed2effect changes direction, giving overall+1/−1/+4/+2pp. This is why first-look wins need complete paired evaluation.

Decision: do not extend this whole grid. Confirm original/drop25and a secondarycontext-mask mechanism contrast using fresh planning seeds1042/1043/1044and strict same-state/RNG600updatetraining repeats. Fit train-onlylinearprobes on the fixed initialscreenweights to inspect physical-state decodability. These use12repeatGPUslots+8validationGPUslots alongside the4selectedshortscreen2jobs. All new training still stops600; no longer candidate run yet.

Toy justification: decoder side information A=C allows a variable code Z=M to reconstruct X=(C,M) while omitting C. This establishes a possible failure mode, not that it caused these PushT results. Anchor dropout may enforce missing scene information, change optimization balance, or simply alter a noisy training trajectory. Independent-case and repeat results will distinguish useful evidence from a selected fluctuation.
