"""Freeze one new TwoRoom validation sample and seven queued native evaluations.

CPU preparation only. Never launches an evaluator, selects by model results,
changes native sampling/balancing, or reads the campaign test split file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

B = Path('/efs/prescottsun/SPT-JEPA')
PYTHON = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
SEED = 2026103101


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def pair_key(pair):
    return (pair['episode_id'], pair['start_step'], pair['goal_step'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=B/'runs/carrier_freshcases_20261031')
    parser.add_argument('--source-record', type=Path, default=B/'runs/carrier_progress_20261031/source-record.json')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for name in ['fresh-manifest.json', 'fresh-selection.json', 'fresh-candidates.json', 'fresh-jobs.json']:
        if (out/name).exists():
            raise FileExistsError(out/name)
    import clear_lewm.manifests as manifests
    import clear_lewm.tasks as tasks
    import clear_lewm.datasets as datasets
    from clear_lewm.protocols import protocol_from_dict

    data = B/'data/hardtasks-v1/tworoom-metadata'
    primary_path = data/'manifests/validation-strict.json'
    primary = read(primary_path)
    card = read(data/'dataset_card.json')
    train = np.load(data/'train_episodes.npy')
    validation = np.load(data/'validation_episodes.npy')
    assert not set(train.tolist()).intersection(validation.tolist())
    assert primary['campaign_split'] == 'validation' and primary['task'] == 'tworoom'
    assert len(primary['pairs']) == 50
    spec = protocol_from_dict(primary['protocol'])
    assert spec.sampling == 'episode-balanced' and spec.goal_offset == 25
    assert all(p['goal_row']-p['start_row'] == spec.goal_offset for p in primary['pairs'])
    assert primary['dataset']['fingerprint']['value'] == datasets.metadata_fingerprint(card['h5_path'])
    blocked = np.asarray(sorted({p['start_row'] for p in primary['pairs']}), dtype=np.int64)
    sampler_audit = {}
    old_read, old_split, old_sample = tasks._read_pair, manifests.split_episode_ids, manifests.sample_rows

    def read_rows(dataset, key, rows):
        # Same h5py order/duplicate compatibility fix as original preparation.
        unique, inverse = np.unique(rows, return_inverse=True)
        return np.asarray(dataset[key][unique])[inverse]

    def exclude_primary_then_native_sample(candidates, episodes, num_eval, seed, mode):
        assert not sampler_audit, 'Native generator must sample exactly once'
        assert num_eval == 50 and seed == SEED and mode == spec.sampling
        candidates = np.asarray(candidates, dtype=np.int64)
        keep = ~np.isin(candidates, blocked)
        admissible = candidates[keep]
        assert len(admissible) >= num_eval
        sampler_audit.update(candidate_count_before_exclusion=len(candidates),
            candidate_count_after_exclusion=len(admissible),
            candidate_rows_sha256_before=hashlib.sha256(candidates.astype('<i8').tobytes()).hexdigest(),
            candidate_rows_sha256_after=hashlib.sha256(admissible.astype('<i8').tobytes()).hexdigest(),
            excluded_start_rows=candidates[~keep].tolist(),
            eligible_episode_count_before=len(np.unique(episodes[candidates])),
            eligible_episode_count_after=len(np.unique(episodes[admissible])),
            eligible_episode_ids_after=np.unique(episodes[admissible]).tolist(),
            admissible_cases=[dict(episode_id=int(episodes[row]), start_row=int(row),
                                   goal_row=int(row+spec.goal_offset)) for row in admissible])
        return old_sample(admissible, episodes, num_eval=num_eval, seed=seed, mode=mode)

    try:
        tasks._read_pair = lambda dataset,key,starts,goals: (read_rows(dataset,key,starts),read_rows(dataset,key,goals))
        manifests.split_episode_ids = lambda episodes,fraction,seed: (train,validation)
        manifests.sample_rows = exclude_primary_then_native_sample
        manifest = manifests.generate_manifest(card['h5_path'],'tworoom',spec,50,SEED,split='heldout')
    finally:
        tasks._read_pair, manifests.split_episode_ids, manifests.sample_rows = old_read, old_split, old_sample
    assert manifest['protocol'] == primary['protocol']
    assert manifest['policy_seed'] == SEED
    keys = {pair_key(p) for p in manifest['pairs']}
    primary_keys = {pair_key(p) for p in primary['pairs']}
    episodes = {p['episode_id'] for p in manifest['pairs']}
    primary_episodes = {p['episode_id'] for p in primary['pairs']}
    assert len(keys) == 50 and not keys.intersection(primary_keys)
    assert episodes.issubset(set(validation.tolist())) and not episodes.intersection(train.tolist())
    assert all(p['cross_room'] and p['start_clear'] and p['goal_clear'] and p['source_window_clean']
               and not p['initial_success'] for p in manifest['pairs'])
    # Preserve native upstream statistics; explicit post-exclusion statistics
    # below clarify that the native counts were measured before our wrapper.
    manifest.update(campaign_split='validation', metadata_card_sha256=sha(data/'dataset_card.json'),
        campaign_scope='New pairs from the existing validation episode pool; not an independent test set. '
                       'Pinned CLEAR task/planner semantics. Released LeWM has unmatched training exposure and capacity.',
        fresh_selection=dict(seed=SEED, primary_manifest=str(primary_path), primary_manifest_sha256=sha(primary_path),
            rule='Exclude only exact primary pairs from native eligible rows before unchanged episode-balanced sampling.',
            candidate_count_after_exclusion=sampler_audit['candidate_count_after_exclusion'],
            episode_overlap_count=len(episodes & primary_episodes), exact_pair_overlap_count=0,
            native_statistics_scope='Original generator counts describe candidates before primary-pair exclusion.'))
    manifest_path = out/'fresh-manifest.json'
    manifests.save_manifest(manifest,manifest_path)
    candidates_path = out/'fresh-candidates.json'
    write(candidates_path,sampler_audit)

    models = []
    for arm in ['identity','bounded']:
        for seed in [1,2]:
            run=B/'runs/ideas_validate600_20261031'/f'tworoom-carrier-{arm}-seed{seed}'
            models.append(dict(id=f'carrier-{arm}-seed{seed}',family='carrier',arm=arm,training_seed=seed,
                               run=run,checkpoint=run/'model-step0000600',released=False))
    for seed in [1,2]:
        run=B/'runs/mechanism_rate600_20261030'/f'tworoom-fresh-rate-on-seed{seed}'
        models.append(dict(id=f'temporal-forecast-seed{seed}',family='temporal-forecast',training_seed=seed,
                           run=run,checkpoint=run/'model-step0000600',released=False))
    models.append(dict(id='released-lewm',family='released-lewm',training_seed=None,
                       checkpoint=B/'weights/hardtasks/tworoom',released=True))
    jobs=[]
    for model in models:
        ckpt=model['checkpoint'];cfg=read(ckpt/'config.json')
        assert (ckpt/'weights.pt').is_file() and (ckpt/'weights.pt').stat().st_size > 0
        if not model['released']:
            assert read(model['run']/'stage-complete.json')['stop_step'] == 600
            expected='as_jepa_lewm.carrier.CarrierLeWM' if model['family']=='carrier' else 'as_jepa_lewm.robust.RobustLeWM'
            assert cfg['_target_'] == expected, (model['id'],cfg['_target_'])
        else:
            assert cfg['_target_']=='stable_worldmodel.wm.lewm.LeWM'
        ident='fresh-tworoom-'+model['id'];dest=out/ident
        jobs.append(dict(id=ident,phase='fresh-validation-planning',task='tworoom',
            family=model['family'],training_seed=model['training_seed'],checkpoint=str(ckpt),output=str(dest),
            checkpoint_config_sha256=sha(ckpt/'config.json'),checkpoint_weights_sha256=sha(ckpt/'weights.pt'),
            checkpoint_weights_bytes=(ckpt/'weights.pt').stat().st_size,source_record=str(args.source_record),
            command=[PYTHON,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(manifest_path),
                     '--checkpoint',str(ckpt),'--data',str(data),'--output',str(dest)]))
    write(out/'fresh-jobs.json',dict(source_record=str(args.source_record),manifest=str(manifest_path),
        manifest_sha256=sha(manifest_path),jobs=jobs,
        scheduling='Prepared only. Parent assigns host/gpu/environment/source and launches after source freeze.'))
    selection=dict(generator_seed=SEED,policy_seed=SEED,num_cases=50,exact_pair_overlap_count=0,
        selected_unique_episodes=len(episodes),primary_unique_episodes=len(primary_episodes),
        episode_overlap_count=len(episodes & primary_episodes),episode_overlap_ids=sorted(episodes & primary_episodes),
        selected_episode_ids=sorted(episodes),primary_episode_ids=sorted(primary_episodes),
        selected_pairs=[[p['episode_id'],p['start_step'],p['goal_step']] for p in manifest['pairs']],
        manifest_sha256=sha(manifest_path),candidates_sha256=sha(candidates_path),
        primary_manifest_sha256=sha(primary_path),helper_sha256=sha(__file__),
        frozen_inputs={str(p):sha(p) for p in [data/'dataset_card.json',data/'train_episodes.npy',data/'validation_episodes.npy',
                          Path(manifests.__file__),Path(tasks.__file__),Path(datasets.__file__)]},
        checks=dict(all_validation=True,no_train_pairs=True,no_test_split_file_access=True,protocol_unchanged=True,
            native_sampler_unchanged=True,no_pair_overlap=True,unique_pairs=True,native_hygiene_passed=True,
            seven_checkpoint_files_verified=len(jobs)==7,no_model_results_used=True,no_gpu_or_training=True),
        scope='Confirmation on new exact pairs in the same validation episode pool, with episode overlap disclosed. '
              'No new training seeds or independent test claim. The pinned generator retains its original dataset-wide '
              'metadata/source-transition-validity computation, but selects only frozen validation episodes. '
              'Policy seed follows native generator convention and changes together with sampling seed, identically for all seven models. '
              'Released LeWM normalization/training exposure and capacity remain those of the existing native reference evaluator.',
        candidate_summary={k:v for k,v in sampler_audit.items() if k!='admissible_cases'})
    write(out/'fresh-selection.json',selection)
    print(json.dumps(dict(output=str(out),manifest_sha256=sha(manifest_path),jobs=len(jobs),
        episode_overlap=selection['episode_overlap_count'],candidate_count=sampler_audit['candidate_count_after_exclusion'],
        eligible_episodes=sampler_audit['eligible_episode_count_after'],checks=selection['checks']),indent=2),flush=True)


if __name__=='__main__':
    main()
