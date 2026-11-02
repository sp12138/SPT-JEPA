"""Operational checkpoint I/O only: use the installed atomic backend and await async completion."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import torch


def safe_save(progress, trainer, module):
    from lightning.pytorch.plugins.io.async_plugin import AsyncCheckpointIO
    from lightning.pytorch.plugins.io.torch_plugin import TorchCheckpointIO
    from as_jepa_lewm.factory import model_target
    io = trainer.strategy.checkpoint_io
    assert isinstance(io, (TorchCheckpointIO, AsyncCheckpointIO))
    assert getattr(TorchCheckpointIO.save_checkpoint, '_spt_atomic_patched', False)
    assert trainer.is_global_zero
    # The backend already writes/fsyncs a unique sibling temp and atomically renames it.
    # An extra fixed last.ckpt.tmp followed by immediate rename is unsafe under AsyncCheckpointIO.
    trainer.save_checkpoint(progress.root / 'last.ckpt')
    if isinstance(io, AsyncCheckpointIO):io.teardown()  # waits; this implementation recreates the executor on next save
    checkpoint = progress.root / f'model-step{trainer.global_step:07d}'
    checkpoint.mkdir(exist_ok=True)
    torch.save(module.model.state_dict(), checkpoint / 'weights.pt')
    (checkpoint / 'config.json').write_text(json.dumps(dict(_target_=model_target(progress.cfg), **progress.cfg['model']), indent=2))
    import shutil
    for older in sorted(progress.root.glob('model-step*'))[:-2]:
        if int(older.name.split('step')[1]) not in progress.cfg.get('retain_steps', []):shutil.rmtree(older)
    record = progress.root / f'safe-checkpoint-io-{trainer.global_step:07d}.json'
    value = dict(step=int(trainer.global_step), checkpoint_io=type(io).__module__ + '.' + type(io).__name__,
        async_completion_waited=isinstance(io, AsyncCheckpointIO), atomic_backend=True,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='I/O path/synchronization only; unchanged trainer checkpoint payload,model,loss,data,optimizer,schedule and sample cursor.')
    record.write_text(json.dumps(value, indent=2) + '\n')


if __name__ == '__main__':
    import as_jepa_lewm.train as entry
    entry.PersistentProgress.save = safe_save
    sys.argv[0] = 'as_jepa_tools.train_stage'
    runpy.run_module('as_jepa_tools.train_stage', run_name='__main__')
