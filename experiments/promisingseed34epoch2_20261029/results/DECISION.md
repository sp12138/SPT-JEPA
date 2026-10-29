# Two more diagnostic seeds at two epochs

All 58 planned jobs exited zero. All four continued full checkpoints pass finite model/optimizer/scheduler, exact 24,982-update sample cursor, RNG presence, and portable/full tensor equality audits.

| Method | Seed3 primary / additional P0 | Seed4 primary / additional P0 |
| --- | ---: | ---: |
| Original beta .25 | 22.00 / 16.00% | 76.00 / 72.00% |
| Retuned beta .0625 | 86.67 / 82.67% | 70.67 / 68.00% |
| Temporal-forecast | 92.67 / 89.33% | 88.00 / 85.33% |

Temporal-forecast improves both diagnostic seeds at the same two-epoch budget. Full per-case evidence, paired conditional intervals and generation diagnostics are retained alongside this decision. These are not third-epoch results, independent final benchmark confirmation, or academic novelty evidence. The original seed3 reaches 86% after three epochs: a large two-epoch gap partly reflects learning speed and must not be presented as the mature gap.

The 2026-10-30 user redirect prioritizes harder control tasks. Keep this campaign held; do not add third-epoch PushT training. Preserve all weaker runs and checkpoints.
