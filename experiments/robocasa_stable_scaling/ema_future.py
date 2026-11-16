"""Predict a slowly moving future representation learned from the same logs."""
import copy
import types

import torch


def encode_with_future_teacher(self, batch):
    pixels = batch['pixels']
    b, w = pixels.shape[:2]
    # prepare_pixels already normalizes RGB. Only the observed initial frame
    # enters the trainable encoder and its shared decoder anchor.
    with torch.autocast(pixels.device.type, enabled=False):
        tokens = self.encoder(pixels[:, 0].float(), interpolate_pos_encoding=True).last_hidden_state
        initial = self.pool(tokens).flatten(-2)
        self.future_encoder.eval(); self.future_pool.eval()
        with torch.no_grad():
            tokens = self.future_encoder(pixels[:, 1:].flatten(0, 1).float(),
                interpolate_pos_encoding=True).last_hidden_state
            future = self.future_pool(tokens).flatten(-2).reshape(b, w-1, -1)
        actions = self.action_encoder(batch['action'].float())
    return dict(emb=torch.cat((initial[:, None], future), 1), act_emb=actions)


@torch.no_grad()
def update_future_teacher(self):
    decay = self.future_teacher_decay
    for student, teacher in [(self.encoder, self.future_encoder), (self.pool, self.future_pool)]:
        for target, source in zip(teacher.parameters(), student.parameters()):
            target.mul_(decay).add_(source, alpha=1-decay)
        for target, source in zip(teacher.buffers(), student.buffers()):
            target.copy_(source)
    self.future_teacher_updates.add_(1)


def install(model, cfg):
    assert cfg.get('raw_pixel_encoder') and cfg.get('raw_shared_visual_anchor')
    assert cfg.get('fp32_encoder') and cfg.get('pixel_objective') == 'future-mean'
    assert not cfg.get('raw_physical_joint') and not cfg.get('joint_views')
    assert not cfg.get('randomize_target_views')
    model.ema_future_targets = True
    model.future_teacher_decay = float(cfg['future_teacher_decay'])
    assert 0 < model.future_teacher_decay < 1
    model.future_encoder = copy.deepcopy(model.encoder).eval().requires_grad_(False)
    model.future_pool = copy.deepcopy(model.pool).eval().requires_grad_(False)
    model.register_buffer('future_teacher_updates', torch.zeros((), dtype=torch.int64))
    model.encode = types.MethodType(encode_with_future_teacher, model)
    model.update_future_teacher = types.MethodType(update_future_teacher, model)
