# AS-JEPA: status, insights and next-round plan (2026-10-29)

Update 2026-10-31: the Phase 1 and Phase 2 plans below are superseded by
[as-jepa-round2-spec.md](as-jepa-round2-spec.md), which adapts blocks A to F to the
collaborator's later full-data PushT results. The insights in Section 2 still hold.

This synthesizes the 2026-10-28/29 review of the collaborator's results and the
design discussion that followed. Evidence and figures are in
[as-jepa-results-review-20261028.md](as-jepa-results-review-20261028.md). Numbers
refer to the collaborator's branch `research/ideas-20261028` at `094c28c`; the
redundant `insights/12h-20261027` branch has been deleted from the remote (all its
commits are contained in the research branch).

## 1. Where we stand

- **A working model exists on one seed, but not the model the spec describes.** Small L2,
  50k steps, `beta = 0.01` (100 times below the spec grid), `sigma0 = 0.1`, a one-time
  initialization fix. Seed 0: h7 agent error 0.23 of copy, action identification 88%,
  P0 69 to 70%, confirmed on temporally disjoint test windows.
- **It does not reproduce reliably.** Seeds 0 and 2 work; seed 1 has a non-empty state but
  weak action following (48%) and planning (13%). The three-seed gate fails on action
  identification (74.2% against 75%) and the 32-case human gallery is unreviewed.
- **Against LeWM.** L2 46.7% ± 29.9 across training seeds; L0 64.2% ± 9.1. Without the failed
  seed L2 is 63.7%, level with L0, but dropping only L2's worst seed is biased; dropping the
  worst of each gives L0 68.3% against L2 63.7%, within planner noise. L2 uses about 8 times
  the GPU time and also renders and follows actions.
- **S line.** S0 and S1 free rollouts are worse than copying (pixel feedback). S2 has no valid
  run (empty state at `beta = 4`; NaN at low `beta`; half-learning-rate retries running).
- **The price has not been shown to help.** Lower `beta` was better at every value tried; no
  matched L1 or near-zero-price control exists at 50k.

## 2. Insights to carry forward

1. **No stop-gradients.** L2 and S2 losses keep the target `mu_t` and the prior's inputs
   attached; all detaches are in logged metrics (`as_jepa_lewm/model.py`,
   `runtime/vforge/models/as_jepa/model.py`).
2. **The current design is an autoencoder plus a priced predictability term.** The past shapes
   the encoder only through `beta * surprise`. Proposition 1 makes this exact: at `beta = 1`
   information pays only if the past predicts it; below 1 the objective rewards information
   for its own sake (`J_beta = J_1 - (1 - beta) * Rate`).
3. **The working model sits in the autoencoder regime.** All 64 dimensions active, about 6 to 10
   nats each, about 500 to 600 nats per window (the story expects about 40), dot colours
   stored, and 128 dimensions also all active and much worse (P0 10%). Consistent with true
   `beta < 1`. Standard PushT cannot tell an autoencoder state from a dynamics state, since
   almost everything beyond the anchor is pose; the dot and colour variants can.
4. **No proposition is contradicted.** Appearance goes to the anchor in two seeds (anchor swap
   moves background colour 94.5% toward the donor, state swap 0.1%); dynamics are decodable
   (probe R² 0.87 agent, 0.97 block); the collapse barrier is real. Transient exclusion
   (Proposition 3) is untested because no working run is known to be at true `beta >= 1`.
5. **The threat is reachability, not falsity.** If purity holds only at prices where end-to-end
   training collapses, the method cannot deliver its main guarantee. The full 10-epoch grid,
   `sigma0`, anchor distance, calibration (I01) and the `lambda` family below all address it.
6. **Initialization must be stated in noise units.** A random encoder maps PushT frames to
   nearly the same vector (spread 0.0007 against `sigma0 = 0.1`), so the renderer sees noise
   and ignores the state. Spread equal to `sigma0` (`u = mu / sigma0 ≈ 1`) worked; spread 1
   was crushed by the initial surprise bill. Rule: start the state's spread across frames at
   about one `sigma0`. L1's BatchNorm fix addresses the same underlying problem.
7. **BF16 can fake a live state.** Several runs had feature spread 0.15 to 0.2 in BF16 and about
   1e-4 in FP32. Collapse tests, probes and rates must run in FP32.
8. **Separate entry and surprise prices equal the `sigma0` knob.** In noise units the entry price
   relative to surprise is `sigma0^2`. A smaller `sigma0` lowers the collapse barrier (which
   comes from entry) and favours slow components.
9. **`beta = 0` is not a usable control.** The surprise term is the prior's only loss in the
   current design (the renderer never reads the prior's output in training), so at zero the
   prior never trains. Use `beta = 1e-3` for "almost no price".
10. **Training budget changes conclusions.** The same run went from P0 24% at 24k to 70% at 50k.
    Short screens must not declare a setting dead.
11. **Roll out in state space.** I08 showed SPT's drift comes mainly from re-encoding generated
    frames; S2's design avoids this. L2 rollouts drift after about 7 steps because the prior's
    multi-step predictions drift (block error 5.1, 10.7, 18.9 px at h1, h7, h13 from forecasts,
    about 5 px from the frame's own state at every horizon).
12. **Anchor distance raises the demand for information in the state** (the renderer cannot
    copy the pose from a distant anchor), but the pull toward predictable states still comes
    from the price. g0 equals gap 8 at `beta = 0.01`; the theory predicts the gap matters near
    the barrier.
13. **The `lambda` family (new theory, sketch level).** Train the renderer on the encoded state
    and, with probability `lambda`, on a sample of the prior's forecast:

    ```
    J*(beta, lambda) = C + beta * Rate - sum_t [ (1 - lambda) I(x_t; s_t | x_0) + lambda I(x_t; ŝ_t | x_0) ]
    ```

    - every `lambda` is a valid code length (negative ELBO of a coin-mixture model);
    - Proposition 2 holds for every `lambda`; Proposition 3 becomes exclusion at `beta >= 1 - lambda`;
    - on `beta + lambda = 1` the autoencoder incentive vanishes; `lambda = 1` is the past-future
      information bottleneck with the anchor as side information;
    - causally, `lambda > 0` trains the interventional quantity used for planning and enforces
      an invariant observation mechanism; the ends coincide under deterministic dynamics;
    - the forecast must be a sample with `sigma0` noise, not the noiseless mean (I05 currently
      uses means), or action information reaches the renderer unpriced;
    - open: A5 and the collapse barrier under `lambda`; the scheduled-sampling bias (Huszár
      2015) under stochastic transitions must be stated.
14. **Planning reads the encoder means and prior rollouts** (LeWM's CEM, GoalMSE, no renderer),
    so P0 tests state geometry and dynamics, not image quality.
15. **Single-seed comparisons are weak.** Seed spread on P0 is ±30 points and on h7 agent/copy
    ±0.25; only the moving-block ratio is stable (±0.02). Use three seeds and stable metrics.

## 3. Decisions

| Decision | Status |
| --- | --- |
| Focus PushT work on the L line; return to S for general video | agreed |
| Test `sigma0` (entry price) against the end-to-end threat | agreed |
| Test the alternative (`lambda = 1`) and the mixture (`lambda = 0.5`) next round | agreed |
| Remove the redundant insights branch | done |
| `lambda` in the paper: in the objective (method and theory), its value in the empirical design | recommended; **your call** |
| Let the running S2 half-learning-rate retries finish; launch no new S work | recommended |
| Initialization rule in noise units and FP32 diagnostics become part of the method and spec | recommended |

## 4. Plan

### Phase 0: no GPU (now)

| # | Action | Owner | Output | Done when |
| --- | --- | --- | --- | --- |
| 0.1 | Commit the results review, figures and this plan to `lingjing/anchored-state-jepa` | us | commit | pushed |
| 0.2 | Message the collaborator: narrow the campaign (keep the full 10-epoch grid, L2 seeds 3/4, I08 seeds 1/2, S2 retries; drop I10; pause other candidates and new S work); state the next-round arms | us, then collaborator | message | collaborator acknowledges |
| 0.3 | Review the 32-case gallery for L2 seed 0 (`/efs/prescottsun/SPT-JEPA/reports/completion-qualitative-review`) | you | pass/fail per case | 32 marked; gate needs 24 |
| 0.4 | Theory note: proofs of Propositions 1λ, 2 (with `lambda`) and 3λ; extended A4 | me | `docs/lingjing/theory_checks/` note | proofs checked |
| 0.5 | Extend the A5 toy (`theory_checks/a5_value_condition.py`) to `lambda`: does the collapse barrier rise or fall? | me | numbers and a plot | sign known |
| 0.6 | Read existing training logs for the entry/surprise split in the 50k runs | collaborator or me | table | known which term dominates |
| 0.7 | Spec amendment: arms below, frozen predictions, FP32 diagnostics, initialization rule, sampled forecasts | me, approved by you | spec section | approved before launch |

0.5 can change Phase 1: if `lambda` raises the barrier sharply, pair the `lambda` arms with
`sigma0 = 0.03`.

### Phase 1: next GPU round (L2, collaborator executes)

Defaults: 50k steps, FP32 diagnostics, initialization spread = `sigma0`, the existing
evaluation package. About 4.6 GPU-hours per 50k run.

| Block | Arms | Seeds | Runs | Question |
| --- | --- | --- | --- | --- |
| A. Does the price help? | L1 at 50k; L2 at `beta = 1e-3` | 0, 1, 2 | 6 | does `beta = 0.01` beat both? |
| B. Entry price | `sigma0` ∈ {0.1, 0.03} × `beta` ∈ {0.01, 0.04, 0.16, 0.64} | 0 | 7 new | does the highest surviving `beta` rise at 0.03? |
| C. Renderer input | `lambda` = 0.5 and 1 (one-step sampled forecast); `lambda` = 0.5 (multi-step sampled forecast) | 0, 1, 2 | 9 | does `lambda` cut drift and keep control? |
| D. Purity test | dot data, `lambda` ∈ {0, 0.5, 1} | 0 | 3 | does dot information vanish at `lambda = 1`? |
| E. Appearance check | colour data, `lambda` ∈ {0.5, 1} | 0 | 2 | unchanged across `lambda`? |
| F. Anchor distance (conditional) | gap ∈ {0, 8, max} × the three highest prices that train in B | 0 | up to 9 | does the surviving `beta` rise with the gap? |

Total about 27 runs without F (about 125 GPU-hours), 36 with F. Also read when they land:
the full 10-epoch `beta` grid, L2 seeds 3/4, I08 seeds 1/2, the S2 retries.

**Frozen predictions (record before launch):**
- D: dot colour R² falls toward chance at `lambda = 1` for any `beta`; if it falls at a
  smaller `lambda*`, true `beta` is roughly `1 - lambda*`.
- E: background-colour information in the state is unchanged across `lambda` (Proposition 2).
- C: the forecast-versus-own-state gap at h7 and h13 shrinks with `lambda`; dynamics probes stay.
- B: lower `sigma0` raises the highest `beta` that passes the collapse test.

**Primary metrics:** moving-block ratio, the forecast-versus-own-state gap at h7/h13, action
identification, collapse test, rate per dimension. P0 is secondary until three seeds exist.

### Decision rules after Phase 1

| Result | Then |
| --- | --- |
| A: `beta = 0.01` does not beat L1 and `beta = 1e-3` | the price does not drive the working model; the story must say so; rely on B, C and calibration to reach the regime where it should matter |
| B: a higher `beta` survives at `sigma0 = 0.03` | adopt 0.03; rerun the dot test at the highest surviving `beta` |
| B: no change | the barrier is not from entry; inspect the surprise term early in training |
| C/D: `lambda > 0` keeps control and removes dot information | adopt `lambda` in the objective; choose its value in the empirical design; write Propositions 1λ to 3λ into the story |
| D: dot information survives at `lambda = 1` | check the sampled-forecast implementation; the mechanism is not operating, so `lambda` loses its main theoretical justification |
| C: `lambda > 0` collapses or loses action following across seeds | keep `lambda = 0`, or use `lambda` only with multi-step forecasts as a drift fix |
| Full 10-epoch grid: a price of at least 1 survives | retest purity claims there; this is the theory's regime |
| Full grid: every spec price collapses | make calibrated `beta` (I01) or `lambda` part of the canonical method |
| Seed 1 failure recurs in seeds 3/4 | diagnose instability (curves, FP32 feature scale, per-dimension rate) before scaling |

### Phase 2: after Phase 1

- Choose the canonical objective (`beta`, `sigma0`, `lambda`) and update the story (§3.2,
  §4.2 "bits chargeable" wording, §4.7, §5 propositions, §8 empirical design) and the spec.
- Three-seed final configuration on full data and the 10-epoch budget; gate check (rollouts,
  action identification, gallery) and P0 against L0.
- Decide on multi-step prior training in latent space if drift remains.
- Port the recipe (initialization rule, price range, `sigma0`, `lambda`, FP32 diagnostics) to
  S2 when moving beyond PushT.
