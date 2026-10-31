# Factual planner-context diagnostic at update2000

All12 frozen models completed successfully. Every reported horizon beats both the fixed training-center zero-state predictor and the corresponding last-real-state copy. This is a result at2000 updates; the earlier600-step cold-h7 failure is a different checkpoint, not contradicted by this assay.

The table reports endpoint MSE divided by the actual target zero-state error. Cold5 ends at obs5; warm5 and cold7 end at obs7. All values below1 beat the fixed training-center predictor. Existing planning is shown only as context: these64 factual windows are not the50 planning cases, and no planning was rerun.

|Task|Seed|Model|Cold1+5, obs5|Warm3+5, obs7|Cold1+7, obs7|Old primary/fresh %|
|---|---:|---|---:|---:|---:|---:|
|tworoom|1|identity|0.489|0.464|0.606|60/60|
|tworoom|1|static|0.476|0.452|0.580|64/64|
|tworoom|1|state|0.643|0.782|0.968|68/72|
|tworoom|2|identity|0.509|0.484|0.624|56/56|
|tworoom|2|static|0.477|0.451|0.582|62/62|
|tworoom|2|state|0.707|0.682|0.929|52/52|
|cube|1|identity|0.524|0.578|0.589|24/16|
|cube|1|static|0.511|0.567|0.570|14/14|
|cube|1|state|0.469|0.491|0.562|14/12|
|cube|2|identity|0.523|0.571|0.582|20/10|
|cube|2|static|0.508|0.550|0.576|16/8|
|cube|2|state|0.496|0.506|0.574|4/16|

At the common obs7 endpoint, warm3+5 has lower point-estimate error than cold1+7 for all12 models. Conditional paired-window95% intervals are entirely negative for all six TwoRoom models and the two Cube STATE models; they include zero for Cube identity/STATIC. This combines two extra true observations with two fewer autonomous prediction steps. It does not isolate the effect of history length alone.

STATIC slightly improves factual error versus identity on both tasks. STATE is worse on TwoRoom but better on Cube, including both genuine planner information patterns. The Cube warm5 five-horizon mean MSE differences STATE−identity are −0.0626 [−0.0828,−0.0419] and −0.0573 [−0.0781,−0.0363]. Nevertheless its existing primary planning14/4% is below identity24/20%, and fresh12/16% versus16/10% is mixed. STATIC also improves Cube factual prediction while both old primary scores decline. Conversely TwoRoom STATE seed1 has worse factual error but higher old planning. Therefore accounting for warm real history does not make factual latent MSE a reliable ordering of these existing planners. This is a descriptive counterexample to using factual MSE alone as a success surrogate, not identification of the cause of planning failure.

No counterfactual candidates, goal-cost ranking, physical readout or new simulator transition was evaluated. The observed/predicted composition is the native factual rollout under recorded controls. Training-center zero is the existing train-calibrated coordinate origin, not a validation-fitted mean or physical zero. Per-horizon raw MSE, copy/zero denominators, prediction energy, alignment, window-paired differences and bootstrap intervals are preserved in results.json; the compact review preserves all12 models. Intervals condition on these fixed models and windows and do not capture independent training-seed uncertainty.

Execution audit: original identity model source32fad0c and STATIC/STATE d6d0a9f remained separate; helper source62a8022. All cached embeddings/actions/windows, source/config/checkpoint hashes and authoritative row/split mappings passed. Original LeWM.rollout is used directly with prefilled embeddings and zero-element shape-only pixels; forbidden encoder/renderer hooks were installed. Full parameters/buffers and Python/NumPy/Torch CPU/CUDA RNG hashes are unchanged. Exact GPU calls:1632 predict,288 action_encoder,0 encoder,0 renderer; no RGB/simulator/physical-label access. The separate actual first-model CPU contract used1 predict/1 action_encoder.

New native cold7 is numerically close, not bitwise identical, to all12 old caches: maximum MSE4.73e−13 and max coordinate difference3.87e−5. The old audit action encoder processes an eighth unused placeholder block, whereas native rollout only encodes the five/seven required blocks; differently shaped numerical operations need not be bitwise identical. New comparisons uniformly use the same-round native rollouts; original caches were preserved. This observation is not a claim that the particular rounding source was independently isolated.

All12 assigned GPUs were verified idle after completion and the exact owned dispatchers/monitors were closed. HOLD_DISPATCH is set on the new root; previous holds and model sources remain unchanged. The sum of twelve timed GPU assay sections was6.73s, with peak allocated memory≤0.251GiB; source verification/model loading was outside those timed sections. No follow-up training or planning was launched.

Artifacts: B/runs/planner_context2000_20261031 (frozen specs, original-cache provenance, per-model predictions and evidence); B/reports/planner-context2000-20261031/{results.json,summary.json,review.json}. B=/efs/prescottsun/SPT-JEPA.
