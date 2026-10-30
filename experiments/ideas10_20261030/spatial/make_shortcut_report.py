"""Render the completed shortcut metrics without rerunning fits."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    data=json.loads(Path(args.input).read_text())
    lines=['# Ordered-query shortcut audit', '',
           'This is a CPU audit of the proposed target, not training of TF/LeWM. Separate existing train/validation episode sets; each episode contributes one 8-frame window and 8 randomly permuted image-prototype queries. Prototype frame indices/timestamps are not classifier inputs. The readouts use identical query assignments in every control.', '',
           'Features: raw 35-step logged actions; three fixed 8×8 RGB query prototypes; optionally the initial fixed 8×8 RGB image. Thus “state-blind” means no initial observation, not no visual information: query prototypes themselves contain future appearances. Each whole/half target has its own readout, trained simultaneously as separate output columns.', '',
           'Fixed classifiers: linear ridge and Gaussian-kernel ridge; ridge coefficient 1 throughout. Feature normalization uses training rows only. Gaussian bandwidth is the median positive squared training-pair distance, without tuning. Adding initial features changes the input dimension and kernel geometry; this is not a capacity-matched functional proof or an exhaustive nonlinear information test.', '',
           '| Task / classifier / inputs | Train combined | Val whole | Val first half | Val second half | Val combined | Val macro recall (mean3) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    parts=['whole','first_half','second_half']
    for task,row in data.items():
        train_majority=[]
        val_majority=[]
        for part in parts:
            counts=row['train_class_counts'][part]
            selected=max(range(5),key=counts.__getitem__)
            train_majority.append(counts[selected]/sum(counts))
            counts=row['validation_class_counts'][part]
            val_majority.append(counts[selected]/sum(counts))
        cells=[100*sum(train_majority)/3]+[100*v for v in val_majority]+[100*sum(val_majority)/3,30.]
        lines.append('| '+task+' / training-majority constant | '+' | '.join(f'{value:.2f}%' for value in cells)+' |')
        for name,model in row['readouts'].items():
            val=model['validation']
            train=sum(model['train'][part]['accuracy'] for part in parts)/3
            macro=sum(val[part]['macro_recall'] for part in parts)/3
            cells=[100*train]+[100*val[part]['accuracy'] for part in parts]+[100*val['combined_accuracy'],100*macro]
            lines.append('| '+task+' / '+name+' | '+' | '.join(f'{value:.2f}%' for value in cells)+' |')
    lines += ['', '| Task / classifier | Initial image gain (combined) | Episode bootstrap 95% CI | Whole / first / second gain |',
              '|---|---:|---:|---:|']
    for task,row in data.items():
        for name,effects in row['paired_initial_state_effect'].items():
            value=effects['combined']
            lo,hi=value['episode_bootstrap_95ci']
            gains='/'.join(f"{100*effects[part]['delta_accuracy']:+.2f}" for part in parts)
            lines.append(f"| {task} / {name} | {100*value['delta_accuracy']:+.2f}pp | [{100*lo:+.2f}, {100*hi:+.2f}]pp | {gains}pp |")
    lines += ['', 'Bootstrap resamples entire validation episodes (5,000 fixed-seed draws), retaining all queries and segments within each episode. Training/validation identities and per-class recalls remain in the JSON. The intervals quantify this fixed-fit paired readout comparison; they do not include retraining-seed uncertainty.', '',
              'A positive initial-image increment supports state information being useful to this readout, not that a new world-model loss must encode or use it. A missing increment leaves the state-forcing claim unsupported; it does not prove the initial image contains no information. Actions and future query prototypes can already leak substantial task progress in expert trajectories. Native-world-model interventions and actual planning remain required.']
    Path(args.output).write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
