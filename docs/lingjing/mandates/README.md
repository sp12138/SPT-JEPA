# Direction mandates for the three shared sessions

Written 2026-11-21. The project has three directions, and Peng has opened one agent
session for each. Each session owns its direction end to end: it plans the runs,
tunes, monitors, analyses and writes up. The work sits with these sessions because
the datasets, checkpoints, Python environment and evaluation tools all live on
Peng's machines and storage, and because the sessions know this code base.
Lingjing's side stays responsible for the method's design principles, answers
design questions, and reviews results. Lingjing's cluster holds a working copy of
the PushT pipeline and is used for independent replications.

| Direction | Mandate | Machine |
| --- | --- | --- |
| A. The original framework (state, predictor and pixel renderer) | [original-framework.md](original-framework.md) | dev13 |
| B. The pixel-free variant | [pixel-free.md](pixel-free.md) | dev14 |
| C. The evaluation framework | [evaluation.md](evaluation.md) | dev3 |

Background for all three: [../../results-index.md](../../results-index.md).

## Rules common to the three directions

1. **Peng's `AGENTS.md` stays in force.** If a step conflicts with it, do the steps
   that do not, and ask Peng in the session about the one that does.
2. **Where to work.** On your machine, under `/efs/prescottsun/SPT-JEPA/`, in your
   own source clone and your own campaign directory. The Mac checkout is shared by
   the three sessions and holds Peng's uncommitted edits: do not edit it. Stay on
   your machine; if you need more GPUs, say so in your status file.
3. **Never touch a job you did not launch.**
4. **You decide** hyperparameters, schedules, the order of experiments, diagnostics,
   engineering and the use of your machine. You do not need to ask before a run
   that stays inside your mandate.
5. **Write before you launch.** Before each batch of runs, add to your status file:
   the question, the arms, what each outcome would mean, the rule by which you will
   decide, and the GPU-hours. A batch above 150 GPU-hours needs a written
   checkpoint first. A knob is searched only when you can state why no principle or
   earlier result fixes its value.
6. **You monitor your own runs.** Chain evaluations to training with detached
   coordinators. While anything of yours is running, check it at least once an
   hour, in the way you did for Peng's hourly checks. Restart an operational
   failure from its saved state and record it; do not restart a run whose loss
   became non-finite, report it. Read the large arrays sequentially once so they
   are in the page cache before starting loaders on a cold dataset.
7. **Evidence.** Two training seeds at least behind any statement, each with its
   own initial weights. The same planning seeds and budgets on both sides of a
   comparison. Negative results are kept and reported with the same care as
   positive ones. Cases you look at repeatedly are development cases.
8. **What comes to Lingjing's side for agreement:** a change to the terms of an
   objective beyond what the mandate lists, a new loss, anything that adds pixels to
   direction B, an EMA target, augmentation, a stop-gradient used as a collapse
   fix, a pretrained component in a from-scratch model, and the final choice of a
   canonical recipe. Write the question in the status file under "Questions for
   Lingjing's side" and continue with other work in the mandate while you wait.
9. **Status file.** `STATUS.md` at your campaign root. Top: the current state in at
   most 15 lines (what is established, what is running, what is next, open
   questions). Below: a dated log with the pre-launch notes of rule 5, result tables
   with file paths, and the decisions you took and why. Update it at every
   milestone. Lingjing's side reads it on its own schedule.
10. **Stop conditions** are in each mandate. When one is met, write the closing
    report and stop; do not extend the search on your own.
