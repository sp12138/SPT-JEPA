# Action-ranking screen: retain all outcomes, do not continue the grid

All120endpoints exit0 and all20full/portable checkpoint audits pass. Same51,097,134parameters, four paired existing2epochtraining seeds,600extra updates. No candidate provides a consistent planning improvement across seeds.

| Setting | Seed1 P0 | Seed2 P0 | Seed3 P0 | Seed4 P0 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original |79|47|30|85|60.25|
| One-step coefficient0.0001 |77|54|25|80|59.00|
| One-step coefficient0.001 |76|47|34|84|60.25|
| Terminal coefficient0.0001 |71|49|27|88|58.75|
| Terminal coefficient0.001 |67|41|20|56|46.00|

One-step-small improves onlyseed2(+7pp), while worsening the others. One-step-large merely matches the aggregate mean; a selected single-seedgain is insufficient. Terminal-small improvesseed4(+3pp) but worsensseed1/3. Terminal-large worsens everyseed(−12/−6/−10/−29pp). Stop allfourcandidate settings at600, preserve the failedstrongvariant, and do not extend this grid.

These alternatives come from reversed/shifted within-video actions. They are not simulator-proven different outcomes, so the loss may price causally equivalent actions as negatives. The result does not disprove all action-contrastive methods. It shows that this parameter-free sample-local prior-energy auxiliary, at these strengths/budget, does not solve the stability problem.

The next targeted hypothesis is decoder side information: a changing code can support conditional reconstruction while omitting scene information supplied by the old image anchor. This could matter when planning uses only the latent code. Compare original with0.1/0.25/0.5sample-windowanchor dropout and observed-context-only anchor masking, fourpairedexistingseeds,600updates. Retain original FM/rate gradients. This is a hypothesis, not an established cause or noveltyclaim.

Keep all pause/completion states distinct. Existing shortscreen2jobs remain active on4separateGPUslots; their matched2epochresult is still pending. Mature-checkpoint adaptation does not establish from-scratch stable89%performance.
