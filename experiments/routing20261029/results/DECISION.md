# Four-seed gradient routing: decision and mechanism evidence

All120endpoints exited0. At exactly600additionalupdates, no candidate establishes stable89%or robust improvement across allfour existing training seeds. This is adaptation of already trained2epochweights, not from-scratch reproduction.

| Routing | Seed1 P0 | Seed2 P0 | Seed3 P0 | Seed4 P0 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original |77|50|32|81|60.00|
| Target stop |78|56|31|81|61.50|
| Input stop |73|44|26|83|56.50|
| Both transition paths stop |76|55|33|78|60.50|
| Renderer-to-encoder stop |3|5|3|2|3.25|

Target-stop is a weak lead (+1/+6/−1/0percentagepoints), not a confirmed winner. Do not automatically run this whole grid longer. Input-stop worsens3of4seeds; defer. Both-rate-paths-stop is mixed; defer. Preserve the weaklead for a focused confirmation if subsequent evidence justifies it.

The strongest result is the negative control: removing only renderer-to-encoder gradients destroys planning and action identification in allfour seeds within600updates. At fixed initial weights/randomness, its forward loss is exactly the original loss; prior/action/renderer parameter gradients are initially identical. The interventions target the gradient path. Later optimizer trajectories and representation/renderer adaptation evolve jointly, so this is evidence for the role of persistent RGB grounding in this configuration, not a universal theorem.

Despite this functional deterioration, allfour renderer-stop checkpoints pass the declared coarse posterior collapse rule: posterior/anchor ratios0.723/0.716/0.643/0.683remainbelow0.9. Action identification is25/21.875/29.6875/25%, near4-waychance, and h7agent/copy is1.125/0.966/1.041/0.944. The collapse gate alone misses control-relevant failure. Report health, generation and planning separately.

Numerical trajectory variation also matters: original-control seed1/2 repeats score77/50here versus78/53in the prior matched600update screen. Equal checkpoints/RNG do not make GPU training bitwise deterministic. A few percentage points in one seed are insufficient for an algorithm claim.

The next bounded screen tests within-video action ranking using the actual prior and the same Euclidean latent goal cost, retaining RGB/rate gradients and51,097,134parameters. One-step and5-stepterminal variants each use two fixed strengths; the original control plus fourpairedexistingseeds occupy20slots. All stop at600updates. Reversed/shifted action alternatives are surrogate negatives, not simulator-proven different outcomes; auxiliary fitting may fail to improve control, as historical I09 already demonstrated. No novelty or performance claim is made.
