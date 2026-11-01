"""A bounded exposure control from positive600 evidence;old roots stay held."""
import argparse
import copy
import gc
from pathlib import Path
import shutil

import torch

from common import B, dataset, read, same, sha, source_check, write
from as_jepa_lewm.train import TrainModule, WindowSampler
from experiments.ideas_validate_20261031.preflight import checkpoint_report
from vforge.ops import OpsConfig, initialize_ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    root = Path(args.root); assert not root.exists()
    parent = B / 'runs/unified_spatial_checkpoint600_20261101'
    assert (parent / 'HOLD').exists()
    old = read(parent / 'protocol.json'); source_check(old['source'])
    evidence = read(B / 'reports/unified-three-20261101/spatial-factorial600.json')
    cube = next(x for x in evidence['paired'] if (x['task'], x['contrast'], x['seed']) ==
                ('cube', 'spatial_minus_flat_ordinary', 'two_fixed_models'))
    assert evidence['passed'] and len(evidence['rows']) == 24
    assert cube['mean_pp'] == 6. and cube['ci95_pp'][0] > 0
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa_math', allow_fallback=False),
                   device=torch.device('cpu'), parallel_state=None)
    runs, proofs = [], []
    for row in old['runs']:
        ident = row['id']; previous = parent / ident; run = root / ident
        terminal = read(parent / 'worker-records' / (ident + '-gate.json'))
        audit_path = parent / 'audits' / (ident + '-step600') / 'audit.json'
        prior_audit = read(audit_path)
        assert terminal['passed'] and prior_audit['passed']
        assert prior_audit['native_cpu_model_optimizer_scheduler_restore']
        checkpoint_path = previous / 'full-archives/step0000600.ckpt'
        assert sha(checkpoint_path) == prior_audit['full_checkpoint_sha256']
        cfg = copy.deepcopy(read(previous / 'config.json'))
        cfg['output'] = str(run)
        assert cfg['model_class'] == 'spatial_native'
        write(run / 'config.json', cfg)
        shutil.copy2(checkpoint_path, run / 'last.ckpt')
        assert sha(run / 'last.ckpt') == sha(checkpoint_path)
        for name in ['initial-model', 'model-step0000600']:
            (run / name).symlink_to(previous / name, target_is_directory=True)
        (run / 'full-archives').mkdir()
        (run / 'full-archives/step0000600.ckpt').symlink_to(checkpoint_path)
        checkpoint = torch.load(run / 'last.ckpt', map_location='cpu', weights_only=False)
        portable = torch.load(run / 'model-step0000600/weights.pt', map_location='cpu', weights_only=True)
        report = checkpoint_report(checkpoint, portable, cfg, 600)
        assert all(report['checks'].values())
        torch.manual_seed(cfg['seed']); module = TrainModule(cfg)
        initial = torch.load(run / 'initial-model/weights.pt', map_location='cpu', weights_only=True)
        assert same(module.model.state_dict(), initial)
        module.load_state_dict(checkpoint['state_dict'], strict=True)
        configured = module.configure_optimizers()
        optimizer, scheduler = configured['optimizer'], configured['lr_scheduler']['scheduler']
        optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
        assert same(module.state_dict(), checkpoint['state_dict'])
        assert same(optimizer.state_dict(), checkpoint['optimizer_states'][0])
        assert same(scheduler.state_dict(), checkpoint['lr_schedulers'][0])
        assert checkpoint['campaign_samples'] == 600 * 128
        train = dataset(cfg, 'train')
        sampler = iter(WindowSampler(train, start=checkpoint['campaign_samples'],
                                     seed=cfg['seed'], batch_size=128))
        assert [next(sampler) for _ in range(16)] == prior_audit['next16_sampler_indices']
        proof = dict(passed=True, actual_post_copy_CPU_model_optimizer_scheduler_restore=True,
                     parent_audit=str(audit_path), parent_audit_sha256=sha(audit_path),
                     config_sha256=sha(run / 'config.json'), checkpoint_sha256=sha(run / 'last.ckpt'),
                     next16_sampler_exact=True, no_GPU_or_forward_or_optimizer_update=True)
        write(root / 'copy-restores' / (ident + '.json'), proof); proofs.append(proof)
        # Composite proof explicitly reuses the actual old GPU backward and
        # binds the new actual CPU restoration. Do not relabel it a new GPU run.
        composed = dict(prior_audit, config_sha256=sha(run / 'config.json'),
                        actual_GPU_backward_reused_from=str(audit_path), post_copy_CPU_restore=proof)
        write(root / 'audits' / (ident + '-step600') / 'audit.json', composed)
        write(root / 'worker-records' / (ident + '-gate.json'), dict(terminal,
              actual_600_worker_reused_from=str(parent / 'worker-records' / (ident + '-gate.json'))))
        runs.append(dict(row, run=str(run), gpu=5 + row['seed'], config_sha256=sha(run / 'config.json')))
        del module, configured, optimizer, scheduler, checkpoint, portable, initial, train, sampler
        gc.collect()
    for task in ['pusht', 'tworoom', 'cube']:
        write(root / 'cohorts' / (task + '.json'), read(parent / 'cohorts' / (task + '.json')))
        for seed in [1, 2]:
            link = root / 'graphs' / f'{task}-seed{seed}'; link.parent.mkdir(exist_ok=True)
            link.symlink_to(B / 'runs/unified_three_20261101/graphs' / link.name, target_is_directory=True)
    write(root / 'protocol.json', dict(old, runs=runs, parent=str(parent),
          note='Six unchanged full-capacity spatial controls600→1200→2000 on ALL3 tasks/seeds. '
               'Cube paired600+6pp[1,12] permits this bounded exposure check,not a long-training claim. '
               'Old parents remain held. Same publicL2/NGC with all case sets retained;no automatic>2000.'))
    write(root / 'source-record.json', old['source'])
    assert not torch.cuda.is_initialized()
    write(root / 'maturity-approval.json', dict(passed=True, run_ids=[r['id'] for r in runs],
          stages=[1200, 2000], helper_source=old['source'], maturity_worker_sha256=sha(args.worker),
          original_CPU_and_GPU600_audits_reused=6, actual_new_post_copy_CPU_restores=6,
          proofs=proofs, evidence_sha256=sha(B / 'reports/unified-three-20261101/spatial-factorial600.json'),
          automatic_long_training=False, continuous_monitor=False))
    print('Prepared6 bounded spatial2000 controls with actual native post-copy CPU restores')


if __name__ == '__main__':
    main()
