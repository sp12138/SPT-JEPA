# Temporal-forecast: origin and contribution boundary

Temporal-forecast is a method combination proposed and implemented during this campaign. It combines I03 (temporally correlated posterior noise) and I05 (forecast-conditioned renderer training) from the [ten-candidate draft](../ideas20261028/IDEAS_DRAFT.md), introduced in commit 9573e9b. The matched-size implementation and factorial controls were introduced in commit 6b83290; see [RobustLeWM](../../as_jepa_lewm/robust.py).

Lingjing's [original specification](../../docs/lingjing/as-jepa-experiment-spec.md), lines 491 and 713, explicitly requires the renderer to read the posterior sample rather than the prior output. I05 deliberately extends this design: half of renderer conditions are causal multi-step prior forecasts, and the image generation loss propagates through those forecasts to the predictor and past encoder. I03 uses posterior noise correlation rho=0.8 with the conditional Gaussian KL correction and unchanged marginal sigma=0.1.

The resulting method retains 51,097,134 model parameters and a sample-level objective. Forecast generation adds training computation; equal parameter count and update count are not equal FLOPs.

[Three-epoch evidence](../promising20261029/results/epoch3/DECISION.md) gives primary P0 94%/94% for training seeds 1/2, with original beta=.25 at 88%/74.67% and retuned beta=.0625 at 86%/84%. [New held-out query sets](newcase-results/DECISION.md) give 91.33%/94%. These are exploratory two-training-seed results, not a broad stability or out-of-distribution claim.

The [component comparison](../promising20261029/results/components/DECISION.md) gives forecast-render alone 90%/90% at two epochs versus 88%/90% for the combination. It does not establish that I03 is necessary. Confirmation therefore prioritizes the simpler I05 path and additional independent training seeds. No academic priority or novelty claim is established by repository provenance or these experiments.
