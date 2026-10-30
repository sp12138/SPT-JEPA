# Local/global1024: bounded2000-update maturity review

**Cube local1024 has a real matched-budget planning improvement, and the600-step failure was not a reliable maturity verdict.** Its strict success rises from0/4% to24/20%, exceeding same-budget64-D identity Carrier8/16% and TF12/6%. This is a promising candidate, not high absolute performance, pose recovery, broad seed stability or a generally superior algorithm. TwoRoom still favors the older64-D Carrier.

| Task | Arm | Success600→2000, seeds1/2 | Clean rollout / initial-copy MSE at2000 | One-step / observed-copy MSE at2000 |
|---|---|---|---|---|
| TwoRoom | local1024 | 16/24 →60/56% | .239/.246 | .183/.198 |
| TwoRoom | dense1024 | 8/6 →10/12% | .370/.367 | .235/.244 |
| Cube | local1024 | 0/4 →24/20% | .361/.354 | .435/.423 |
| Cube | dense1024 | 4/10 →16/12% | .499/.482 | .532/.519 |
| TwoRoom |64-D identity Carrier |44/30 →72/70% | .216/.196 | .171/.171 |
| Cube |64-D identity Carrier |4/6 →8/16% | .194/.185 | .292/.294 |

At2000 the other64-D Carrier arms score TwoRoom74/82% (free chart),76/80% (bounded); Cube10/10%,8/14%. TF2000 scores TwoRoom2/6%, Cube12/6%. These are actual case-audited2000 controls, unlike the historical scalar-only TF600 context.

Direct local−dense differences are TwoRoom+50/+44pp, mean+47pp, shared-case95% interval[34,60]; Cube+8/+8pp, mean+8pp[0,16]. Cube local−identity64 is+16/+4pp, mean+10pp[2,19]; local−TF64 is+12/+14pp, mean+13pp[4,22]. TwoRoom local−identity64 remains−12/−14pp, mean−13pp[−27,0]. Intervals condition on two trained seeds and resample the same50 cases jointly; they exclude training-seed uncertainty and multiplicity adjustment. Both seeds and all comparisons remain in the JSON.

## What matured

All eight models' observed-state probe outputs are **exactly identical to their600-step outputs**, as expected from unchanged frozen encoders, projection buffers and calibration. The observation representation did not improve during this continuation. For Cube local, causal rollout/copy falls from.774/.765 to.361/.354. Swapping its initial state now multiplies rollout MSE by3.41/3.39; using another episode's actions multiplies it by2.65/2.75; changing action order gives1.98/1.98. TwoRoom local corresponding ratios are4.47/4.38,3.78/3.75 and2.55/2.50. Initial state and executed actions therefore materially contribute to learned outcomes.

The renderer also starts using the learned condition. At t=.95, Cube local wrong-state/correct-forecast denoising error ratios become1.57/1.68, wrong-action ratios1.32/1.35, and initial-carry ratios1.78/1.83. At600 these interventions were nearly ineffective. TwoRoom local ratios are1.32/1.21 and1.14/1.10 for wrong state/actions. Dense also develops conditioning sensitivity; larger sensitivity alone does not make its goal metric or planning better.

This supports a maturation explanation for the original short gate: additional native world-model learning made an unchanged fixed measurement more useful to both prediction and planning. It does not isolate the contributions of predictor learning versus renderer gradients, and the common1200 resume boundary is part of the recorded continuation. No component-specific causal attribution is claimed.

## Remaining limits and research implication

Cube local still has a *larger* own-normalized latent prediction error than64-D Carrier despite better planning. Predictability of a chosen representation is not identical to accuracy in physically relevant distinctions; ratios normalized within different representation spaces are not a common physical metric. This is compatible with retaining useful detail that is harder to predict, but the present experiment does not prove that information detail causally mediates the planning gain. Local/dense differ in spatial averaging, projection subspace and metric, rather than merely invertible coordinate rotation.

There is no plateau evidence yet. Comparing mean1500–1700 with1800–2000, Cube local rate decreases8.1/8.6% and flow MSE12.0/14.2%; dense rate decreases6.1/5.7% and flow MSE14.2/14.0%. TwoRoom local rate decreases22.3/23.3% and flow MSE20.3/24.6%. The maturity check therefore supports retaining Cube local as a candidate rather than rejecting it from600 results. It does not itself authorize another stage; all runs remain capped2000 for root review.

Observed clean1024-D ridge probes retain their earlier conditioning/generalization problems; the predefined noisy probe can outperform the clean probe. No probe or normalization was retuned. No new physical forecast/pose readout was computed, so these latent errors do not establish centimeter/degree future accuracy. The separate TF6000 position/pose audit showed TF's advantage over fixed64-D Carrier already in observed **position** readability, while neither had a reliable pose-readout advantage. That evidence prevents attributing all remaining Cube failures solely to pose or solely to forecasting; its6000-budget results must not be merged into this2000-budget comparison.

## Integrity and compute

All64 endpoints exited0, including all16 native full-state audits/archives at1200 and2000. Each confirms exact step/cursor, finite model/optimizer/scheduler, strict native model and optimizer/scheduler load roundtrips, saved RNG, sampler continuation, portable/full-state equality, and all210 actual frozen parameters/projection/mean/scale tensors unchanged. Original600 pre/post-copy hashes and RNG/sampler records agree. Source32fad0c remains the training and GPU-evaluation source; only CPU preparation/archive helpers use0934d3. All config semantics, calibration hashes, strict50 cases, CEM configuration and64-window diagnostics match the600 parent; matched2000 controls share data/exposure/update budget and planner cases/settings.

Local and dense have equal learned capacity and initialization. The1024-D models have55.91/55.98M total parameters versus about51.10M in64-D controls, plus different state/rate dimension, preprocessing and inference cost. Dense's fixed projection adds192MiB versus local12KiB. These comparisons are not capacity/FLOP matched against64-D controls. Recorded50-case evaluation seconds are TwoRoom local43.98/43.40, dense51.08/51.42; Cube local92.41/94.84, dense95.64/98.55. They include setup and variable executed horizons, so lower wall time is not a per-action inference-speed claim.

Full results and provenance: `/efs/prescottsun/SPT-JEPA/reports/local-isotropic2000-20261031/`. Local `summary.json` retains every seed, both training trajectories, physical-probe arrays, intervention measures and paired comparisons. Collection used existing JSON artifacts only; no model forward, fitting, GPU, new assay, model change or automatic continuation occurred.
