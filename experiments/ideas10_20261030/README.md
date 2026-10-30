# Ten mechanism-distinct proposals: evidence screening

New user request, 2026-10-30: produce ten substantively different ideas aimed at large gains on TwoRoom and Cube under the same evaluation conditions. Most must arise from our own mechanism design and pass deep theoretical or short experimental validation, rather than restating published methods. Absolute global novelty and future benchmark gains cannot be certified in advance; distinguish supported mechanisms, unresolved novelty, and unmeasured task performance.

This campaign first records independent designs, then audits their mathematics and nearest primary literature. A proposed title or analogy does not count as a mechanism. Previously proposed I01–I10, temporal-forecast, ordinary spatial pooling, noise shifts and known metric fixes cannot be recounted as new ideas. Preserve all historical run sources, checkpoints and holds.

Admission requires: an explicit sample-level learning mechanism; no optimal-action or privileged-state training labels; a precise claim with assumptions and a falsification; an executable check or a nontrivial derivation; a credible connection to both tasks; substantive comparison with the closest existing algorithms; and separation from the other nine candidates. A synthetic identity is not task-performance evidence. Literature familiarity is not proof of novelty.

Fairness contract: retain the current data splits and strict validation manifests, observation inputs and action access, model/planner budgets and success semantics. Any proposed change in inference cost, auxiliary queries, data access or memory must be charged and reported. Released LeWM weights share the evaluator but not a demonstrated identical training budget. Physical labels may be used for diagnostics, never silently for candidate learning.

Root coordinates dev2/dev13/dev15 and any bounded GPU work. Sub-agents may work independently on design, primary literature and CPU falsification. Every new training implementation must preserve the existing trainer/checkpoint machinery, freeze a committed source/config, pass execution checks, and stop around600 updates for review before longer work. Do not launch a blanket long grid merely to fill GPU slots.

Evidence labels: **T** = a proved mechanism property within stated assumptions; **C** = controlled numerical/synthetic falsification; **D** = frozen real-data diagnostic; **S** = short world-model training and matched task evaluation. None implies a future substantial benchmark gain. Rejected candidates remain documented.
