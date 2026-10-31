"""Render the completed, audited factorial result without additional computation."""
import argparse
import json
from pathlib import Path


def interval(x):
    a,b=x['ci95_pp']
    return f"{x['delta_pp']:+.0f} [{a:.0f}, {b:.0f}]"


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=json.loads(Path(a.results).read_text());assert r['passed'] and r['all48exit0']
    names=['white_l2','white_ngc','correlated_l2','correlated_ngc']
    lines=['# Fixed-model NGC × action-correlation control', '',
        'All 48 new correlated-proposal endpoints exited 0 and are held. The 48 completed white-proposal endpoints are retained as the other two cells. No training, graph encoding, additional cases, or parameter selection occurred.', '',
        'The two public controller changes do not show a general additive benefit on Cube. Native improves from 24/29% white-L2 to 36/36% correlated-L2, then only to 37/38% with NGC (primary/fresh). TF changes from 20/15% to 33/31%, then 33/28%; local1024 changes from 19/22% to 23/29%, then 23/22%. All within-correlated Cube NGC-effect intervals include zero. Native has the highest observed Cube mean in all four cells, subject to different capacities and exposure.', '',
        'The clearest interaction contrast is TF on fresh Cube: NGC adds 10pp under white proposals and −3pp under correlated proposals, giving −13pp [−24, −2]. This is evidence against a simple additive account in these fixed runs. It is not proof of a general negative interaction: intervals omit simulator-repeat variance and population training-seed uncertainty, and no multiplicity correction was applied.', '',
        'On TwoRoom, local1024 retains a large NGC benefit under correlated proposals: +39pp primary and +22pp fresh. Correlation itself does not improve its NGC result (95/91% versus white-NGC 98/94%). Local1024 ranks below native under L2 but above native under NGC in both proposal regimes; TF remains weak. This preserves the task-dependent evidence for an evaluation-cost limitation without establishing a new learning-algorithm gain or a single cross-task winner.', '',
        'Each cell evaluates the same frozen checkpoint on the same 50 cases. W is the original diagonal CEM; C is the frozen training-only correlation parameterization. L2 is native GoalMSE; NGC is the public graph cost with the uniform FP32 direct-distance numerical repair. The CEM budget remains 300 candidates × 30 iterations, five model steps, history three and receding horizon five. Correlation preserves the initial per-coordinate marginal scale; later elite distributions can differ.', '',
        '## Two-seed means and paired contrasts', '',
        'Percent success; contrast intervals are case-paired 95% bootstrap intervals conditional on these two trained seeds. Interaction = (C-NGC − C-L2) − (W-NGC − W-L2).', '',
        '| Task | Model | Cases | W-L2 | W-NGC | C-L2 | C-NGC | NGC effect within C, pp | Interaction, pp |',
        '|---|---|---|---:|---:|---:|---:|---|---|']
    for x in r['factorial_combined']:
        v=x['cells'];d=x['contrasts']
        lines.append('| '+' | '.join([x['task'],x['family'],x['case_set'],*[f'{v[k]:.0f}'for k in names],interval(d['correlated_ngc_minus_l2']),interval(d['interaction'])])+' |')
    lines+=['', '## Every training seed', '', '| Task | Model | Seed | Cases | W-L2 | W-NGC | C-L2 | C-NGC |', '|---|---|---:|---|---:|---:|---:|---:|']
    for x in r['factorial']:
        lines.append('| '+' | '.join([x['task'],x['family'],str(x['seed']),x['case_set'],*[f"{x['cells'][k]:.0f}"for k in names]])+' |')
    pairs=r['comparisons'];count=sum(x['first_scores_exact_cases']for x in pairs)
    lines+=['', '## Audit and interpretation limits', '',
        f"All {len(pairs)} within-correlated L2/NGC pairs have identical first-iteration action, goal, and predicted endpoint hashes on all {count} case banks, with identical saved L2/NGC score arrays. Later CEM populations may diverge. White and correlated physical candidate banks deliberately differ; their Gaussian innovation equivalence is supported by the frozen CPU contracts, not an unrecorded cross-run tensor equality claim.", '',
        f"All 48 complete model state hashes are unchanged. Actual model imports, card scalers, checkpoint/source/graph/case/config provenance and fixed Cholesky matrix hashes pass ({r['immutable_inputs_checked']} unique immutable inputs checked). The first returned action-map error is zero in every run; mapped objective metadata is checked by the runtime wrapper. The twelve graph artifacts are reused unchanged.", '',
        'The white-proposal cells are the immediately preceding completed evaluations of these same checkpoints/cases, not simultaneous repeated evaluations in the new correlated batch. White-L2 historical replay differences remain separate: '+str(r['historical_replay']['exact'])+'/24 exact case vectors. No historical or contemporary best-of baseline is selected. The repeated simulator protocol is not outcome-bitwise reproducible; paired intervals do not include this repeat variability.', '',
        'All models use ordinary recorded-action conditioning. The public controller changes use no physical-state or optimal-action labels. NGC uses 5,000 training-only observation nodes; the correlation matrix uses the fixed 128 training clips. These are controller controls, not new sample-level learning objectives.', '',
        'Capacity/exposure differ across model families: local1024 is approximately 56M parameters at 6,000 updates on both tasks; native is approximately 18M with width192 at 6,000 TwoRoom / 12,000 Cube; TF is approximately 51M with width64 at the same respective budgets. The local family is local1024 on both tasks, not a task-specific replacement with F64.', '',
        r['limitations'], '',
        'Runtime includes both-score instrumentation in both cost arms, so it is not bare L2 latency. Per-case outcomes, all five factorial contrasts and conditional intervals are retained in the local results.json; full candidate bank metadata and arrays remain on EFS.', '']
    Path(a.output).write_text('\n'.join(lines))


if __name__=='__main__':main()
