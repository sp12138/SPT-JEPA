# Mature Cube: identical old candidate-bank comparison

All cells are seed1/seed2;50 paired cases. Original frozen scorer ea11036; no new simulator, fitting, renderer or training.

## Primary:22 nonoracle

|Model|Physical↔encoded rank|Encoded↔predicted rank|Encoded hold %|Predicted hold %|Encoded joint %|Predicted joint %|Encoded pose cost|Predicted pose cost|Metric regret|Signed dynamics regret|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|native12000|0.321/0.313|0.837/0.827|22.000/24.000|20.000/20.000|28.000/32.000|18.000/22.000|2.822/2.600|3.218/2.792|0.728/0.506|0.396/0.193|
|native23342|0.323/0.315|0.841/0.839|24.000/22.000|20.000/26.000|30.000/28.000|16.000/30.000|2.608/2.548|3.235/2.872|0.514/0.454|0.627/0.324|
|tf12000|0.346/0.355|0.951/0.949|14.000/18.000|16.000/14.000|20.000/24.000|22.000/26.000|3.631/3.076|3.421/3.126|1.537/0.982|-0.210/0.051|
|tf23342|0.352/0.355|0.961/0.962|18.000/18.000|20.000/16.000|24.000/24.000|26.000/22.000|2.940/2.948|3.124/3.448|0.846/0.855|0.185/0.500|

Shared coverage: hold 40.0%, endpoint joint 46.0%; physical endpoint-cost selector pose 2.0938, hold 40.0%. This selector is not a hold-success ceiling.

## Appendix:17 recorded-action-neighborhood

|Model|Physical↔encoded rank|Encoded↔predicted rank|Encoded hold %|Predicted hold %|Encoded joint %|Predicted joint %|Encoded pose cost|Predicted pose cost|Metric regret|Signed dynamics regret|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|native12000|0.901/0.912|0.897/0.890|56.000/56.000|58.000/56.000|100.000/100.000|94.000/98.000|0.013/0.013|0.512/0.437|0.000/0.000|0.498/0.424|
|native23342|0.909/0.913|0.941/0.933|56.000/56.000|58.000/60.000|100.000/100.000|98.000/100.000|0.013/0.013|0.157/0.156|0.000/0.000|0.144/0.143|
|tf12000|0.876/0.882|0.911/0.900|56.000/56.000|54.000/46.000|100.000/98.000|98.000/98.000|0.013/0.040|0.244/0.253|0.000/0.026|0.230/0.213|
|tf23342|0.884/0.886|0.933/0.936|56.000/56.000|56.000/58.000|100.000/100.000|100.000/100.000|0.013/0.013|0.083/0.098|0.000/0.000|0.070/0.085|

Shared coverage: hold 82.0%, endpoint joint 100.0%; physical endpoint-cost selector pose 0.0135, hold 56.0%. This selector is not a hold-success ceiling.

## Appendix:all39

|Model|Physical↔encoded rank|Encoded↔predicted rank|Encoded hold %|Predicted hold %|Encoded joint %|Predicted joint %|Encoded pose cost|Predicted pose cost|Metric regret|Signed dynamics regret|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|native12000|0.728/0.736|0.919/0.914|56.000/56.000|48.000/50.000|100.000/100.000|82.000/88.000|0.013/0.013|0.898/0.629|0.000/0.000|0.885/0.616|
|native23342|0.740/0.738|0.931/0.933|56.000/56.000|52.000/56.000|100.000/100.000|92.000/92.000|0.013/0.013|0.247/0.289|0.000/0.000|0.233/0.275|
|tf12000|0.671/0.693|0.947/0.948|56.000/56.000|26.000/26.000|100.000/98.000|44.000/70.000|0.013/0.040|2.556/1.169|0.000/0.026|2.542/1.129|
|tf23342|0.696/0.703|0.963/0.965|56.000/56.000|48.000/50.000|100.000/100.000|86.000/92.000|0.013/0.013|0.636/0.223|0.000/0.000|0.623/0.210|

Shared coverage: hold 82.0%, endpoint joint 100.0%; physical endpoint-cost selector pose 0.0135, hold 56.0%. This selector is not a hold-success ceiling.

## Paired changes (23342−12000)

|Model seed|Group|Metric|Mean delta|95% conditional case interval|
|---|---|---|---:|---:|
|native23342-seed1|nonoracle|encoded_predicted_rank|0.0042|[-0.0220,0.0339]|
|native23342-seed1|nonoracle|predicted_hold_success|0.0000|[-0.0800,0.0800]|
|native23342-seed1|nonoracle|predicted_pose_cost|0.0167|[-0.3593,0.3824]|
|native23342-seed1|nonoracle|encoded_oracle_hold_success|0.0200|[0.0000,0.0600]|
|native23342-seed1|nonoracle|encoded_oracle_pose_cost|-0.2144|[-0.5437,-0.0065]|
|native23342-seed1|nonoracle|metric_regret|-0.2144|[-0.5437,-0.0065]|
|native23342-seed1|nonoracle|signed_dynamics_regret|0.2311|[-0.1672,0.6683]|
|native23342-seed2|nonoracle|encoded_predicted_rank|0.0121|[-0.0164,0.0445]|
|native23342-seed2|nonoracle|predicted_hold_success|0.0600|[0.0000,0.1400]|
|native23342-seed2|nonoracle|predicted_pose_cost|0.0795|[-0.2714,0.4976]|
|native23342-seed2|nonoracle|encoded_oracle_hold_success|-0.0200|[-0.0800,0.0400]|
|native23342-seed2|nonoracle|encoded_oracle_pose_cost|-0.0515|[-0.3085,0.1415]|
|native23342-seed2|nonoracle|metric_regret|-0.0515|[-0.3085,0.1415]|
|native23342-seed2|nonoracle|signed_dynamics_regret|0.1311|[-0.2932,0.6283]|
|tf23342-seed1|nonoracle|encoded_predicted_rank|0.0099|[0.0056,0.0143]|
|tf23342-seed1|nonoracle|predicted_hold_success|0.0400|[-0.0400,0.1200]|
|tf23342-seed1|nonoracle|predicted_pose_cost|-0.2963|[-0.8008,0.1084]|
|tf23342-seed1|nonoracle|encoded_oracle_hold_success|0.0400|[-0.0400,0.1200]|
|tf23342-seed1|nonoracle|encoded_oracle_pose_cost|-0.6911|[-1.3101,-0.1718]|
|tf23342-seed1|nonoracle|metric_regret|-0.6911|[-1.3101,-0.1718]|
|tf23342-seed1|nonoracle|signed_dynamics_regret|0.3948|[-0.2488,1.0665]|
|tf23342-seed2|nonoracle|encoded_predicted_rank|0.0131|[0.0077,0.0189]|
|tf23342-seed2|nonoracle|predicted_hold_success|0.0200|[-0.0800,0.1200]|
|tf23342-seed2|nonoracle|predicted_pose_cost|0.3220|[-0.2157,0.9075]|
|tf23342-seed2|nonoracle|encoded_oracle_hold_success|0.0000|[0.0000,0.0000]|
|tf23342-seed2|nonoracle|encoded_oracle_pose_cost|-0.1270|[-0.3771,0.0000]|
|tf23342-seed2|nonoracle|metric_regret|-0.1270|[-0.3771,0.0000]|
|tf23342-seed2|nonoracle|signed_dynamics_regret|0.4490|[-0.1836,1.2040]|

## Scope

- Fixed old50x39 bank, not current CEM proposal coverage or complete50-step planning success.
- First-plan H1/T5; no warm3+5/fresh/second-segment outcomes.
- 22 nonoracle is primary;17 recorded-action-neighborhood and all39 reported separately.
- Physical labels are diagnostic only. Endpoint joint and anywhere-during-trajectory hold3 are distinct.
- Metric plus signed-dynamics regret is an algebraic selector identity, not a unique causal decomposition; signed term can be negative.
- Bootstrap resamples50 paired cases, not candidates or independent training seeds.
- Old reset-vs-dataset RGB mismatch and one-realization simulator variability remain; no new simulation.
- Raw native BN and original raw commands retained; no common-coordinate raw latent MSE comparison.

Full per-case arrays and file provenance remain in EFS results.json. The small summary retains all8 model/budget rows, all3 groups and all paired intervals.
