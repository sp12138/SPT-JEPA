"""Fixed1024-D addressed/global carrier controls; native TF objective unchanged."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from as_jepa_lewm.robust import RobustLeWM

PROJECTION_SEED=20261030
PATCH_SIDE=16
CHANNELS=192
STATE_DIM=1024
DENSE_BLOCK=128


def projection_matrix(mode):
    """Exact projection rules from the completed CPU local1024 diagnostic.

    Private NumPy streams do not consume global NumPy/Torch initialization RNG.
    Persist the resulting fixed buffer to make exported checkpoints self-contained.
    """
    if mode=='local':
        rng=np.random.default_rng(np.random.SeedSequence([PROJECTION_SEED,CHANNELS,16]))
        return (rng.standard_normal((CHANNELS,16)).astype(np.float32)/np.float32(4.))
    if mode!='dense':raise ValueError('spatial_mode must be local or dense')
    dim=PATCH_SIDE*PATCH_SIDE*CHANNELS
    result=np.empty((dim,STATE_DIM),dtype=np.float32)
    for block,start in enumerate(range(0,STATE_DIM,DENSE_BLOCK)):
        rng=np.random.default_rng(np.random.SeedSequence([PROJECTION_SEED,dim,STATE_DIM,block]))
        value=rng.standard_normal((dim,DENSE_BLOCK),dtype=np.float32)
        value*=np.float32(1/math.sqrt(STATE_DIM))
        result[:,start:start+DENSE_BLOCK]=value
    return result


class SpatialCarrierLeWM(RobustLeWM):
    """Same64×16 native state interface in both fixed-measurement arms.

    No learned chart/head, auxiliary loss, privileged label, or batch statistics.
    Compared with the old64-D carrier, state dimension, parameter count, rate
    scale and compute differ. Only local versus dense is the matched control.
    """
    def __init__(self,spatial_mode='local',spatial_mean=None,spatial_scale=None,
                 state_slots=64,state_width=16,**kwargs):
        if spatial_mode not in {'local','dense'}:raise ValueError('Unknown spatial_mode')
        if (state_slots,state_width)!=(64,16):raise ValueError('This gate requires64 slots×16 channels')
        if kwargs.get('mode','L2')!='L2' or kwargs.get('domain','standard')!='standard':
            raise ValueError('This fixed gate requires standard benchmark L2')
        if kwargs.get('encoder_size',224)!=224 or kwargs.get('pool_partition','full')!='full':
            raise ValueError('This audited carrier requires224 pixels and the unchanged unused full pool')
        if kwargs.get('batchnorm_state',False) or kwargs.get('batchnorm_projector',False):
            raise ValueError('Calibration must remain fixed, never batch coupled')
        if kwargs.get('response_query_mode','none')!='none':raise ValueError('No auxiliary head in this gate')
        if (spatial_mean is None)!=(spatial_scale is None):raise ValueError('Provide both fixed statistics or neither')
        super().__init__(state_slots=state_slots,state_width=state_width,**kwargs)
        self.encoder.requires_grad_(False);self.pool.requires_grad_(False)
        self.encoder.eval();self.pool.eval()
        self.spatial_mode=spatial_mode
        self.register_buffer('spatial_projection',torch.from_numpy(projection_matrix(spatial_mode)))
        self._spatial_has_statistics=spatial_mean is not None
        mean=torch.zeros(STATE_DIM) if spatial_mean is None else torch.as_tensor(spatial_mean,dtype=torch.float32)
        scale=torch.ones(STATE_DIM) if spatial_scale is None else torch.as_tensor(spatial_scale,dtype=torch.float32)
        if mean.shape!=(STATE_DIM,) or scale.shape!=(STATE_DIM,):raise ValueError('Require1024 coordinate statistics')
        if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale>0).all()):
            raise ValueError('Finite mean and strictly positive scale required')
        self.register_buffer('spatial_mean',mean.clone());self.register_buffer('spatial_scale',scale.clone())

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();self.pool.eval();return self

    @torch.no_grad()
    def patch_features(self,pixels):
        if pixels.ndim!=5 or pixels.shape[2:]!=(3,224,224):raise ValueError('Expected[B,T,3,224,224] normalized pixels')
        with torch.autocast(device_type=pixels.device.type,enabled=False):
            tokens=self.encoder(pixels.float().flatten(0,1),interpolate_pos_encoding=True).last_hidden_state
            patches=tokens[:,1:].float()
            if patches.shape[1:]!=(256,CHANNELS):raise ValueError('Expected16×16 patches×192 channels')
            return patches.reshape(*pixels.shape[:2],256,CHANNELS)

    @torch.no_grad()
    def project_patches(self,patches):
        if patches.shape[-2:]!=(256,CHANNELS):raise ValueError('Expected256 addressed patches×192 channels')
        leading=patches.shape[:-2]
        with torch.autocast(device_type=patches.device.type,enabled=False):
            if self.spatial_mode=='local':
                # Contiguous2×2 spatial mean, then one shared16-channel map.
                grid=patches.float().reshape(-1,8,2,8,2,CHANNELS).mean(dim=(2,4))
                result=(grid@self.spatial_projection.float()).reshape(*leading,STATE_DIM)
            else:
                result=(patches.float().reshape(-1,256*CHANNELS)@self.spatial_projection.float()).reshape(*leading,STATE_DIM)
        return result

    @torch.no_grad()
    def raw_carrier(self,pixels):return self.project_patches(self.patch_features(pixels))

    def encode(self,info):
        if not self._spatial_has_statistics:raise RuntimeError('Prepare train-only statistics before encode or training')
        info['emb']=(self.raw_carrier(info['pixels'])-self.spatial_mean)/self.spatial_scale
        if 'action' in info:info['act_emb']=self.action_encoder(info['action'])
        return info

    def audit_expected_gradients(self):
        return dict(encoder='frozen',pool='frozen',predictor='active',renderer='active',action_encoder='optional')

    def spatial_parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
            frozen=sum(p.numel() for p in self.parameters() if not p.requires_grad),
            fixed_projection_elements=self.spatial_projection.numel(),
            fixed_projection_bytes=self.spatial_projection.numel()*self.spatial_projection.element_size(),
            fixed_statistics_elements=self.spatial_mean.numel()+self.spatial_scale.numel(),
            state_slots=self.state_slots,state_width=self.state_width,state_dim=self.state_dim)
