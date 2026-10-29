# Same-size robustness experiments

The full-data beta=.25 three-epoch model has useful control but does not pass the predeclared stability screen: new training seeds1-4 average84.0% P0, sampleSD6.28pp, worst74.67%; fresh eval cases83.67%,SD6.50pp. This campaign compares a simpler lower-beta control, I03 temporal posterior noise, I05 forecast rendering, and their factorial combination.

Every arm uses the identical51,097,134-parameter model, full data,64-dimensional state, sigma=.1, batch128,37473 updates and124910-step LR schedule. All objectives remain sample/window-level; no cross-sample regularizer or BatchNorm is added. Forecast unrolling costs extra compute, which will be reported.

See [prespecified protocol](protocol.json). New training seeds5/6 are reserved for a single selected winner, not for repeatedly tuning methods on confirmation results. Full outputs/checkpoints stay under /efs/prescottsun/SPT-JEPA/runs/robustness_20261029. No outcome is available yet.
