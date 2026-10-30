"""Render the CPU mechanism checks, retaining weak and negative observations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parent
    d=json.loads((root/'checks.json').read_text());a=d['action_annihilator']
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    axes[0].loglog(a['scales'],a['smooth_residual'],'o-',label='Smooth free response')
    axes[0].loglog(a['scales'],a['contact_residual'],'s-',label='Contact kink')
    axes[0].set(xlabel='Action scale ε',ylabel='Annihilated signal norm',title='Constructive scaling test')
    axes[0].legend();axes[0].grid(alpha=.2)
    labels=['TwoRoom\nfree proxy','TwoRoom\nblocked proxy','Cube\nno .5 crossing','Cube\n.5 crossing']
    vals=[100*d['real_data'][task]['groups'][event]['median_visual_residual_energy_ratio']
          for task in ['tworoom','cube'] for event in ['False','True']]
    axes[1].bar(range(4),vals,color=['#64748b','#2563eb','#64748b','#2563eb'])
    axes[1].set(xticks=range(4),xticklabels=labels,ylabel='Median residual / centered visual energy (%)',title='Existing data: fixed smooth image measurements')
    axes[1].set_ylim(0,20);axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
    for ext in ['png','svg']:fig.savefig(root/f'action-annihilator-checks.{ext}',dpi=170)
    plt.close(fig)
    lines=['# Spatial proposal CPU checks','',
        'No model was trained and no GPU was used for these checks. They verify mathematical constructions and inspect existing data; they do not establish improved world modeling or planning.','',
        '## Constructive results','',
        f"- Affine annihilation error: {a['affine_annihilation_error']:.3g}; cumulative-coordinate change error: {a['action_coordinate_invariance_error']:.3g}.",
        f"- Smooth-response observed scaling order: {min(a['smooth_observed_order']):.6f}–{max(a['smooth_observed_order']):.6f}; contact-kink order: exactly1 across four halvings.",
        f"- State-dependent affine example: global action-mean residual norm {a['global_action_mean_residual_norm']:.6f}, local action-annihilator norm {a['local_annihilator_residual_norm']:.3g}. The methods have different nullspaces.",
        '- Contact projection: 10,000 known-halfspace pairs satisfy firm nonexpansion to3.56e−15 and idempotence to4.45e−16. This verifies a classical contact-model property; it is not a novelty result.',
        '- First-passage example: AA and BB end in the same state, but hit probabilities are1 and0. Right-censoring gives the true hazard.2; keeping only observed events gives.539823. These are classical identities, not a learned-model result.','',
        '## Existing-data action-annihilator audit','',
        'Same predeclared128 training episodes per task, one eight-frame window each at the existing frameskip5. P=I−[1,U][1,U]† for this original CPU audit. ψ is fixed4×4 cosine coefficients per RGB channel. Physical fields are diagnostic only. The implementation subsequently selected by root uses the more conservative [1,time,U] span and8×8 area-pooled RGB; the following numbers must not be relabeled as measurements of that different target.','',
        '| Task | Diagnostic partition | Windows | Median visual residual energy fraction | Median physical residual norm |',
        '|---|---|---:|---:|---:|']
    for task in ['tworoom','cube']:
        for event in ['False','True']:
            g=d['real_data'][task]['groups'][event]
            lines.append(f"| {task} | {event} | {g['n']} | {g['median_visual_residual_energy_ratio']:.5f} | {g['median_absolute_position_residual']:.6g} |")
    lines += ['',
        'TwoRoom partitions indicate whether any raw displacement differs from nominal5×action by more than1e−3 pixels. This is a dynamics-residual proxy, not a certified contact oracle. Its free group has median position residual amplitude fraction1.53e−7 and absolute norm8.11e−6 pixels, while the blocked/nonaffine group has fraction.11386 and norm7.20 pixels. This supports the intended cancellation mechanism in this dataset.', '',
        'Cube partitions indicate whether the recorded continuous gripper-contact value crosses.5; the physical residual refers to block xyz and is in meters. The event group has a larger absolute block residual (17.18mm versus.156mm Frobenius norm over the clip), but its median visual residual energy fraction is LOWER (.02954 versus.04965). Thus the present visual measurements do not establish contact enrichment for Cube. The relative physical residual fraction is also not larger. No task-success inference follows.', '',
        'The first attempted Cube proxy was any change in the continuous contact field. That was too broad; it was replaced explicitly by fixed.5 crossing, with0 and.9 sensitivity below. This correction is not a training-label or hyperparameter choice. All groups and thresholds are retained.', '',
        '| Cube threshold | No-cross / cross windows | No-cross visual fraction | Cross visual fraction |', '|---:|---|---:|---:|']
    for threshold,g in d['real_data']['cube']['contact_threshold_sensitivity'].items():
        lines.append(f"| {threshold} | {g['False']['n']} / {g['True']['n']} | {g['False']['median_visual_residual_energy_ratio']:.5f} | {g['True']['median_visual_residual_energy_ratio']:.5f} |")
    lines += ['', '## Rank of the selected implementation', '',
        'For X=[1,time,cumulative standardized controls], the same128-window audit yields TwoRoom residual rank4 in all128 windows; Cube rank1 in127 and rank2 in1. No zero ranks occur. The invariant version retains only one temporal contrast for almost all Cube windows. See time-projector-ranks.json and IMPLEMENTATION.md.', '',
        '![Constructive and existing-data checks](action-annihilator-checks.png)', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
