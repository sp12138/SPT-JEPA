"""Official LeWM modules/objective with explicit project data/budget adaptation.

The native loss retains BatchNorm and SIGReg. This is a baseline, not a proposed
sample-level objective. No renderer, residual carry or privileged training input.
"""
import torch
from torch import nn
from stable_pretraining.backbone.utils import vit_hf
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor, Embedder, MLP
from stable_worldmodel.wm.loss import SIGReg


class NativeLeWM(LeWM):
    def __init__(self, window=8, action_dim=10):
        if window != 8 or action_dim not in (10, 25):
            raise ValueError("This matched baseline is window8, TwoRoom10/Cube25 actions")
        encoder = vit_hf(size="tiny", patch_size=14, image_size=224,
                         pretrained=False, use_mask_token=False)
        predictor = Predictor(num_frames=window-1, input_dim=192, hidden_dim=192,
            output_dim=192, depth=6, heads=16, mlp_dim=2048, dim_head=64,
            dropout=.1, emb_dropout=0.)
        action_encoder = Embedder(input_dim=action_dim, smoothed_dim=10,
                                  emb_dim=192, mlp_scale=4)
        projector = MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d)
        pred_proj = MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d)
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        self.sigreg = SIGReg(knots=17, num_proj=1024)
        self.window = window
        self.state_dim = 192
        self.mode = "native_lewm"
        # Existing native trainer records optional renderer size; there is none.
        self.renderer = None
        self.register_buffer("image_mean", torch.tensor([.485,.456,.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("image_std", torch.tensor([.229,.224,.225]).view(1,3,1,1), persistent=False)

    def normalize(self, pixels):
        shape = pixels.shape
        flat = pixels.flatten(0,1) if pixels.ndim == 5 else pixels
        if flat.shape[-2:] != (224,224):
            raise ValueError("BenchmarkWindows must supply the matched RGB224 input")
        flat = (flat.float()/255. - self.image_mean) / self.image_std
        return flat.unflatten(0,shape[:2]) if pixels.ndim == 5 else flat

    def prepare(self, batch):
        # Shared read-only state-audit interface. The loss uses only RGB/actions.
        return (self.normalize(batch["pixels"]), batch["render"].float()/127.5-1.,
                batch["anchor"].float()/127.5-1.)

    def loss(self, batch, step=0, ramp_steps=0):
        if ramp_steps:
            raise ValueError("No loss-coefficient ramp in the native recipe")
        pixels = self.normalize(batch["pixels"])
        # Native lejepa_forward also replaces sequence-boundary NaNs with zero.
        actions = torch.nan_to_num(batch["action"], 0.0)
        output = self.encode(dict(pixels=pixels, action=actions))
        emb, act = output["emb"], output["act_emb"]
        prediction = self.predict(emb[:,:self.window-1], act[:,:self.window-1])
        # Deliberately preserve native autocast arithmetic: no extra float cast,
        # target detach, batch regularizer replacement or optimizer-dependent loss.
        pred_loss = (prediction-emb[:,1:]).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0,1))
        total = pred_loss + .09*sigreg_loss
        return total, dict(loss=total, pred=pred_loss, sigreg=sigreg_loss,
                          feature_std=emb.detach().float().std((0,1)).mean())

    def training_optimizer_config(self, total_steps):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5,
                                      weight_decay=1e-3, betas=(.9,.999))
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
            warmup_steps=max(1,int(.01*total_steps)), max_steps=total_steps)
        return dict(optimizer=optimizer,
                    lr_scheduler=dict(scheduler=scheduler, interval="step"))

    def audit_expected_gradients(self):
        return {name:"active" for name in
                ("encoder", "predictor", "action_encoder", "projector", "pred_proj")}
