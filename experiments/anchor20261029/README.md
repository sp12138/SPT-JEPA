# Sample-level past-anchor information test

Original control plus anchor dropout0.1/0.25/0.5and observed-context-only masking; four existing training seeds,600updates only. Original flow/rate gradients and51,097,134parameters retained. No batch statistics, extra head or inference changes. This tests whether decoder side information permits omission of goal-relevant information; it does not assume that mechanism is already proven.

Conditioning dropout is a known technique. This is an exploratory side-information ablation and not a novelty claim. Distinguish task performance, reconstruction, formal collapse and costs.
