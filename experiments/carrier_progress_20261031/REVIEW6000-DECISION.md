# Reviewed follow-up after all three completed gates

All68 Carrier/TF6000 endpoints, all40 isotropic1024 endpoints and all four calibrated native-LeWM600 planning runs completed successfully. Completed dispatchers were verified idle and closed; previous runs remain held. See the complete, all-seed results in `results/learning6000/RESULTS.md`, `../local_carrier_20261031/results/gate600/RESULTS.md` and `../native_baseline_20261031/results/gate600/RESULTS.md`.

The central counterexample is Cube: fixed-carrier clean rollout/copy MSE improves from approximately.19 to.15, and wrong actions still hurt, yet its planning stays near10–12%. Learned TF improves to24/14% on the same primary cases. Thus better average prediction in a fixed measurement space does not establish better goal-directed control. Prior pose readouts suggest that the metric may poorly resolve important orientation distinctions; they do not prove that the information is absent. TwoRoom benefits strongly from a stable measurement space (free chart90/94%, fresh82/86% versus TF2/4%, fresh4/2%). This is evidence of a task-dependent tradeoff between preserving information and learning useful distinctions, not evidence for a universally superior frozen encoder.

The1024 gate does not overturn this finding. It learns state/action-dependent transitions, and local structure helps TwoRoom relative to dense projection, but Cube local0/4% is worse than dense4/10%. Its relative prediction error is still larger than the64-dimensional model at600, while losses continue decreasing. One bounded maturation comparison is warranted; calling this a winner or launching a long grid is not.

Native LeWM600 has a separate observation-projector BN running-statistics mismatch. Frozen, training-only preciseBN removes the huge common prediction bias and passes single-sample inference checks, but does not provide a consistent planning gain. Keep raw and calibrated deployment results separate.600 updates remain inside the1249-step warmup, so this is not a mature LeWM comparison.

Reviewed allocation:

- Hold all eight Carrier6000 runs, including the strong TwoRoom runs. Do not extend the original Cube RP64 model merely because rendering loss decreases.
- Advance only the two Cube TF controls with actual task improvement from6000 to12000 on dev13 GPUs0/1, preserving8000/10000/12000 exports and the original8e0f821 training source.
- Advance all eight isotropic local/dense controls from600 to1200, require a full-state archive audit, then stop at2000 on dev15 GPUs0–7. Preserve the original32fad0c training source and all task/seed pairs. No subsequent extension is automatic.
- Advance four native LeWM controls from their original full-state600 checkpoints to2000 on dev2 GPUs0–3. Preserve7a1ab4f training and the one-group297-tensor optimizer. Evaluate both raw and separately calibrated exports; never resume training from a calibrated export.

Every new plan needs committed, immutable helper code and successful actual pre-resume audits. These are14 useful training allocations, not a claim that all24 GPUs are occupied. The two-model Cube pose proposal is still only a proposal. Short-stage gains and two training seeds do not establish broad stability or novelty; a sample-level method substantially ahead on both tasks remains an open goal.
