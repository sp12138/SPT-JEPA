# Standing goals for the three shared sessions

Confirmed by Lingjing on 2026-11-22 (decisions in Section 8). These goals replace
the work lists and stop conditions of the mandates in
[../mandates/](../mandates/README.md). The mandates' common rules 1 to 9 stay in
force, except that rule 2 ("stay on your machine") and rule 8 (what needs
agreement) are replaced by Sections 4 and 6 here and by each goal file.

## 1. Why goals replace the mandates

The mandates gave each session a short work list with a stop condition. Two of the
three sessions reached a stop or a block within a day and then sat idle until a
message arrived from Lingjing's side. A standing goal is different in three ways:

1. It names the result the direction is pursuing, as a ladder of milestones, so the
   session always knows what comes after the current batch.
2. It has no stop condition that ends the work. A milestone that cannot be met ends
   with a written account, and the session moves to the next rung or to its backlog.
3. The session's own agent holds and operates the goal. Lingjing's side does not
   drive it with keep-alive messages; it reads the status files, answers the
   questions addressed to it, and reviews milestone reports.

## 2. Who does what

| Role | Who | What |
| --- | --- | --- |
| Worker and operator | the three shared sessions | plan, launch, monitor, tune, debug, analyse and write up everything in their direction, including the queue that keeps their GPUs busy |
| Design owner and reviewer | Lingjing's side | the method's principles, the answers to design questions, the review of milestone reports, the decisions listed as "needs agreement" |
| Machines and house rules | Peng | `AGENTS.md`, machine access, anything about jobs the sessions did not launch |

## 3. The three directions

| | Direction | What it pursues | Goal file |
| --- | --- | --- | --- |
| A | Original framework (state, predictor, pixel renderer) | one model that learns generation and a representation for control together, and performs decently on hard manipulation | [A-original-framework.md](A-original-framework.md) |
| B | Decoder-free variant | a JEPA-family method with no pixel loss and a theoretical account of what its state keeps | [B-decoder-free.md](B-decoder-free.md) |
| C | Evaluation framework | the staged protocol for measuring world models, run systematically over model families and datasets | [C-evaluation.md](C-evaluation.md) |

How they connect. C builds one portable stage-2 kit (policy head, state probes,
closed-loop evaluation loops) and owns the measurement rules. A and B plug their
models into that kit through a small adapter and run it themselves; they do not
write their own heads or probes. A and B in turn give C its controlled model arms.
Until the kit exists, A and B work on the parts that do not need it.

## 4. Machines

The project operates five machines of eight H200 GPUs each, 40 GPUs in total, all
mounting `/efs/prescottsun/SPT-JEPA/`. On 2026-11-22 at 13:50 UTC the five were
dev3, dev6, dev13, dev14 and dev15, all running and reachable.

| Session | Home machines | GPUs | Reason |
| --- | --- | --- | --- |
| A | dev13, dev15 | 16 | renderer training on Meta-World is the heaviest single job type; A also runs its own stage 2 |
| B | dev14 | 8 | its runs are short (about 0.3 GPU-hours each on PushT); it is limited by design, not compute |
| C | dev3, dev6 | 16 | the largest number of jobs: every family, dataset, regime and seed |

**The shared file.** `/efs/prescottsun/SPT-JEPA/runs/lingjing_goals/MACHINES.md`
holds one line per machine: SSH alias, owner session, platform task name, GPUs in
use, GPUs on loan and to whom, and the time of the last check. It is the
authority on which machines are the project's five; the names above are only the
starting state. Session A creates it from the table above when it starts; every
session keeps its own lines current and appends, never rewrites, the others'.

**When a machine goes down.** Lingjing has authorised the sessions to restore
machines by the manual at the root of `main` (`开发机操作手册.md`). The owner of the
machine does it, following the manual's order: inventory first, then repair the SSH
route, wait for a queued request, or rebuild, as its Section 3 table says. Rules
that keep three sessions from colliding:

- Before any platform change, write a line "restoring <alias>, <time>" in
  `MACHINES.md`, and check that no other session has written one for that machine.
- The project holds five machines, never more. A rebuilt machine replaces the one
  that failed, under whatever alias and cluster the manual leads to; update its
  line. Do not open a sixth, and do not touch any machine outside the five,
  including `dev6-2c` and the machines running Peng's other work.
- Restoring a machine and resuming training are separate steps: after the machine
  is back, restart your own jobs from their saved state and record both.
- While your machine is down, move queued work to your other machine or borrow
  lent GPUs; do not wait idle.

**Lending.** A session whose queue is empty lends its free GPUs by writing them in
`MACHINES.md`. A borrower runs only jobs that save their state and can be stopped,
and gives the GPUs back within 15 minutes of the owner writing a reclaim line.
Nobody uses a GPU that is not theirs or on loan to them in that file, and nobody
touches a job they did not launch.

## 5. Keeping the machines busy with useful work

Each session keeps a queue file (`QUEUE.md` at its campaign root) of runs that are
ready to launch: each entry already has its question, arms, decision rule and
GPU-hours written (common rule 5). A detached dispatcher starts the next entry when
a GPU frees up, so occupancy does not depend on the agent being awake.

The queue is filled in this order, and a session never launches a run only to
occupy a GPU:

1. the current milestone's runs;
2. diagnostics and controls that the current milestone's open questions call for;
3. the next milestone's preparation (data caches, baselines, reference rows);
4. the backlog in the goal file (more seeds behind established claims, staged
   evaluation of existing checkpoints, longer training of the current best recipe,
   further scaling points);
5. if all four are empty, lend the GPUs (Section 4) and say so in the status file.

## 6. How a session operates its goal

1. The session's agent sets the goal text of its goal file as its own session goal,
   with its own goal tool. If the agent has no goal tool, it treats the goal file as
   its standing instruction and re-reads it at every status update.
2. The goal is never "complete" before the last milestone. When a milestone needs
   an answer from Lingjing's side, the session writes the question and continues
   with other rungs or the backlog. Only a question to Peng about machine access can
   pause a session, and then only for the affected machine.
3. Once an hour while anything runs: check jobs, restart operational failures from
   saved state, refill the queue, update the status file.
4. After a context compaction: re-read the goal file and the top of `STATUS.md`
   before doing anything else.
5. **Design changes.** A session may explore a principled change without asking,
   including a departure from the main recipe's architecture, when strong evidence
   supports it. Before the first run it writes in `PROPOSALS.md` the evidence and
   the derivation from the method's principle. The constraints each goal file lists
   as fixed stay fixed. Which recipe is named canonical is reviewed by Lingjing's
   side, and that review does not block further work.

## 7. Records, and where Lingjing's side reads them

At each campaign root on the EFS:

| File | Content |
| --- | --- |
| `STATUS.md` | top 15 lines: what is established, what is running and when it ends, what is next, open questions. Below: the dated log of common rule 9 |
| `QUEUE.md` | the ready-to-launch queue of Section 5 |
| `DECISIONS.md` | one entry per decision the session took: date, the evidence, the alternatives, why this one |
| `PROPOSALS.md` | design changes proposed to Lingjing's side: the evidence, the derivation from the method's principle, the screening result, the status (proposed, agreed, declined) |
| `milestones/<id>.md` | the report that closes each milestone: the question, the table with file paths, what is and is not matched in each comparison, the conclusion and what it changes |

**Mirror on GitHub.** After each milestone and at least every six hours, the
session copies these compact files, never raw run records, checkpoints or logs, to
`docs/lingjing/goals/status/<A|B|C>/` on the GitHub branch `status/lingjing-goals`
(created from `consolidate/20261121`), and pushes from a machine that can reach
GitHub (dev13 could on 2026-11-21). Each session writes only its own folder, and
pulls with rebase before pushing. Lingjing's side reads the state of all three
directions from that branch without sending a message. If a push fails, say so in
the chat status and keep the files current on the EFS.

Each session also posts a five-line status in its chat at every milestone: finished
(with numbers), running (with end time), next, question for Lingjing's side,
question for Peng.

## 8. Decisions taken by Lingjing on 2026-11-22

1. **Datasets for A:** the staged measurements go first onto the existing PushT,
   Cube and TwoRoom checkpoints, in parallel with pretraining on Meta-World, which
   is the main benchmark. A small benchmark stays only if it separates the models.
2. **Scope for C:** Meta-World, the RoboCasa goal-image suite, and PushT and Cube
   as pilot; the model families of C's goal file. RoboCasa365, LIBERO-Plus, DROID
   and a from-scratch generative video arm are deferred.
3. **Design freedom:** Section 6, item 5.
4. **Records:** the GitHub mirror of Section 7.
5. **Machines:** five in total, split two, one, two; the sessions may restore
   machines by the manual (Section 4).
