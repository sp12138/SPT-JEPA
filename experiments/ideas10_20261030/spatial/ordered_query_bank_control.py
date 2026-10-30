"""One predeclared independent-bank sampler control. CPU only, no tuning."""
import argparse
import json
from pathlib import Path

import numpy as np

from ordered_query_checks import transfer
from ordered_query_shortcuts import LOOKUP, PARTS, run_task


BANK_SIZE = 64
BANK_SEED = 20261101
QUERY_COUNT = 8
QUERY_SEED = 20261102


def image_feature(pixels):
    pixels = np.asarray(pixels, dtype=np.float32)/255
    return pixels.reshape(8, 28, 8, 28, 3).mean(axis=(1, 3)).ravel()


def build_bank(data, training_episodes, audit_episodes):
    eligible = np.setdiff1d(training_episodes, np.asarray(audit_episodes))
    assert len(eligible) >= BANK_SIZE
    rng = np.random.default_rng(BANK_SEED)
    selected = rng.choice(eligible, BANK_SIZE, replace=False)
    features, provenance = [], []
    for episode in selected:
        frame = int(rng.integers(0, int(data['ep_len'][episode])))
        absolute = int(data['ep_offset'][episode])+frame
        features.append(image_feature(data['pixels'][absolute]))
        provenance.append(dict(episode=int(episode), frame=frame, absolute_index=absolute))
    bank = np.stack(features)
    query_rng = np.random.default_rng(QUERY_SEED)
    triples = np.stack([query_rng.choice(BANK_SIZE, 3, replace=False) for _ in range(QUERY_COUNT)])
    return bank, triples, provenance


def collect(data, split, episode_pool, bank, triples, expected_episodes):
    rng = np.random.default_rng(20261030 if split=='train' else 20261032)
    selected = sorted(rng.choice(episode_pool, min(128, len(episode_pool)), replace=False).tolist())
    assert selected == expected_episodes
    rows = {key:[] for key in ['actions','queries','initial','labels','episodes','constant_word','initial_carry_labels']}
    starts = []
    for episode in selected:
        length = int(data['ep_len'][episode])
        assert length >= 36
        start = int(rng.integers(0, length-35))
        indices = int(data['ep_offset'][episode])+start+5*np.arange(8)
        features = np.stack([image_feature(pixels) for pixels in data['pixels'][indices]])
        actions = np.asarray(data['action'][int(indices[0]):int(indices[-1])], dtype=np.float32).ravel()
        starts.append(dict(episode=int(episode), start=start))
        for triple in triples:
            centers = bank[triple]
            word = ((features[:,None]-centers[None])**2).sum(-1).argmin(-1)
            label = [LOOKUP[tuple(transfer(segment).ravel())] for segment in [word,word[:4],word[4:]]]
            carry = LOOKUP[tuple(transfer((int(word[0]),)).ravel())]
            rows['actions'].append(actions)
            rows['queries'].append(centers.ravel())
            rows['initial'].append(features[0])
            rows['labels'].append(label)
            rows['episodes'].append(episode)
            rows['constant_word'].append(bool(np.all(word==word[0])))
            rows['initial_carry_labels'].append([carry]*3)
    return {key:np.asarray(value) for key,value in rows.items()}, starts


def support(rows):
    return dict(queries=len(rows['labels']), episodes=len(np.unique(rows['episodes'])),
                constant_symbol_word_fraction=float(rows['constant_word'].mean()),
                class_counts={part:np.bincount(rows['labels'][:,i],minlength=5).tolist() for i,part in enumerate(PARTS)},
                actual_initial_symbol_carry_agreement={part:float(np.mean(rows['labels'][:,i]==rows['initial_carry_labels'][:,i])) for i,part in enumerate(PARTS)},
                carry_scope='Uses the observed initial image and the fixed query partition, no later frames or actions; a label diagnostic, not a fitted classifier.')


def report(results, path):
    lines=['# Independent-bank query sampler: one CPU control', '',
        'Predeclared once: 64 training-frame prototypes per task, drawn from 64 training episodes excluded from all 128 audit-training and 128 validation episodes; bank seed 20261101. Eight ordered triples are drawn once with seed 20261102 and reused across every window and split. No query or bank frame was selected using labels. The prototype indices/timestamps are not readout inputs.', '',
        'This adds access to 64 RGB frames per task for the auxiliary query bank; it adds no interaction, action, success label, or validation image to the bank. The same eight triples contain at most 24 of those frames; all 64 accessed bank frames are charged. No GPU, world-model training, tuning, additional seed search, or new idea is involved.', '',
        'The target remains the complete five-class A→B transformation of the whole window and its two halves. Fixed readouts and preprocessing are unchanged: linear/Gaussian-kernel ridge 1; normalization and median kernel bandwidth from training inputs only. Episode cohorts and start times match the prior within-clip audit. Kernel geometry changes when initial features are added, so this is not a strict capacity proof.', '',
        '| Task / split | I / A / B / AB / BA (whole) | I / A / B / AB / BA (halves pooled) | Constant symbol path | Initial-symbol carry agreement (whole / first / second) |',
        '|---|---|---|---:|---|']
    for task,row in results.items():
        for split,s in row['support'].items():
            whole='/'.join(map(str,s['class_counts']['whole']))
            halves='/'.join(str(a+b) for a,b in zip(s['class_counts']['first_half'],s['class_counts']['second_half']))
            carry='/'.join(f"{100*s['actual_initial_symbol_carry_agreement'][part]:.2f}%" for part in PARTS)
            lines.append(f"| {task} / {split} | {whole} | {halves} | {100*s['constant_symbol_word_fraction']:.2f}% | {carry} |")
    lines += ['', '| Task / readout | Majority validation | Queries+actions | +Initial image | Initial increment | Paired episode 95% CI |',
              '|---|---:|---:|---:|---:|---:|']
    for task,row in results.items():
        counts=row['readout_audit']['train_class_counts']
        validation=row['readout_audit']['validation_class_counts']
        majority=np.mean([validation[part][np.argmax(counts[part])]/sum(validation[part]) for part in PARTS])
        for kind in ['linear','rbf']:
            audit=row['readout_audit']
            blind=audit['readouts'][kind+'/queries_actions']['validation']['combined_accuracy']
            state=audit['readouts'][kind+'/queries_actions_initial']['validation']['combined_accuracy']
            delta=audit['paired_initial_state_effect'][kind]['combined']
            lo,hi=delta['episode_bootstrap_95ci']
            lines.append(f"| {task} / {kind} | {100*majority:.2f}% | {100*blind:.2f}% | {100*state:.2f}% | {100*delta['delta_accuracy']:+.2f}pp | [{100*lo:+.2f}, {100*hi:+.2f}]pp |")
    lines += ['', 'The paired intervals resample validation episodes, not individual queries; they do not include training-seed uncertainty. Full per-segment and per-class metrics remain in the JSON, including the unchanged action-only/query-only linear controls.', '',
        '**Interpretation limit:** an independent bank removes the guarantee that every prototype appears, but can make targets mostly constant-region identity. If initial-symbol carry already matches most targets, a larger initial-image increment mainly supports static query grounding; it does not establish action-conditioned path prediction or planning gains. A small increment leaves even this state-use incentive weak. Either outcome is reported without changing bank size, query seed, bandwidth rule or episode selection.', '',
        'This completes the one requested sampler control. No follow-up search or GPU run is implied.']
    path.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--previous',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    import hdf5plugin
    import h5py
    base,out=Path(args.data),Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    previous=json.loads(Path(args.previous).read_text())
    results={}
    for task in ['tworoom','cube']:
        meta=base/(task+'-metadata')
        card=json.loads((meta/'dataset_card.json').read_text())
        train_pool=np.load(meta/'train_episodes.npy')
        validation_pool=np.load(meta/'validation_episodes.npy')
        expected={split:previous[task][split+'_episodes'] for split in ['train','validation']}
        audit_episodes=expected['train']+expected['validation']
        with h5py.File(card['h5_path'],'r',swmr=True) as data:
            bank,triples,provenance=build_bank(data,train_pool,audit_episodes)
            assert not set(p['episode'] for p in provenance).intersection(audit_episodes)
            splits,starts={},{}
            for split,pool in [('train',train_pool),('validation',validation_pool)]:
                splits[split],starts[split]=collect(data,split,pool,bank,triples,expected[split])
                np.savez_compressed(out/(task+'-bank-control-'+split+'.npz'),**splits[split])
        results[task]=dict(protocol=dict(bank_size=BANK_SIZE,bank_seed=BANK_SEED,query_count=QUERY_COUNT,query_seed=QUERY_SEED,
                             bank_frame_access=BANK_SIZE,unique_used_bank_frames=int(len(np.unique(triples))),
                             triples=triples.tolist(),bank_provenance=provenance,windows=starts),
                          support={split:support(rows) for split,rows in splits.items()},
                          readout_audit=run_task(splits['train'],splits['validation']))
        (out/'ordered-query-bank-control.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        print(json.dumps({task:dict(support=results[task]['support'],paired=results[task]['readout_audit']['paired_initial_state_effect'])},indent=2),flush=True)
    report(results,out/'ORDERED_QUERY_BANK_CONTROL.md')


if __name__=='__main__':main()
