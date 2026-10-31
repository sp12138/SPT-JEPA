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
BASIS_SEED=20261031


def orthogonal_basis():
    """One full signed/permuted Hadamard basis, not a Haar random rotation.

    All entries are exactly representable +/-1/32. Private RNG leaves model,
    data and caller initialization streams unchanged. No task or model seed.
    """
    matrix=np.ones((1,1),dtype=np.float32)
    while len(matrix)<STATE_DIM:
        matrix=np.block([[matrix,matrix],[matrix,-matrix]])
    rng=np.random.default_rng(np.random.SeedSequence([BASIS_SEED,STATE_DIM]))
    matrix=matrix[rng.permutation(STATE_DIM)][:,rng.permutation(STATE_DIM)]
    row_sign=rng.choice(np.array([-1.,1.],dtype=np.float32),STATE_DIM)
    col_sign=rng.choice(np.array([-1.,1.],dtype=np.float32),STATE_DIM)
    return matrix*row_sign[:,None]*col_sign[None,:]/np.float32(32.)


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

    Optional shared local chart changes coordinates without mixing spatial slots.
    No auxiliary loss/head, privileged label, or batch statistics.
    Compared with the old64-D carrier, state dimension, parameter count, rate
    scale and compute differ. Only local versus dense is the matched control.
    """
    def __init__(self,spatial_mode='local',spatial_mean=None,spatial_scale=None,spatial_basis='identity',
                 spatial_chart='none',spatial_state_interface='chart',state_slots=64,state_width=16,**kwargs):
        if spatial_mode not in {'local','dense'}:raise ValueError('Unknown spatial_mode')
        if spatial_basis not in {'identity','orthogonal'}:raise ValueError('Unknown spatial_basis')
        if spatial_chart not in {'none','identity','bounded'}:raise ValueError('Unknown spatial_chart')
        if spatial_state_interface not in {'chart','raw'}:raise ValueError('Unknown spatial_state_interface')
        if spatial_state_interface=='raw' and spatial_chart=='none':
            raise ValueError('The raw chart interface requires an explicit identity or bounded chart')
        if spatial_chart!='none' and (spatial_mode!='local' or spatial_basis!='identity'):
            raise ValueError('Shared chart gate requires the original local spatial addresses')
        if (state_slots,state_width)!=(64,16):raise ValueError('This gate requires64 slots×16 channels')
        if kwargs.get('mode','L2')!='L2' or kwargs.get('domain','standard')!='standard':
            raise ValueError('This fixed gate requires standard benchmark L2')
        if kwargs.get('encoder_size',224)!=224 or kwargs.get('pool_partition','full')!='full':
            raise ValueError('This audited carrier requires224 pixels and the unchanged unused full pool')
        if kwargs.get('batchnorm_state',False) or kwargs.get('batchnorm_projector',False):
            raise ValueError('Calibration must remain fixed, never batch coupled')
        if kwargs.get('response_query_mode','none')!='none':raise ValueError('No auxiliary head in this gate')
        if (spatial_mean is None)!=(spatial_scale is None):raise ValueError('Provide both fixed statistics or neither')
        if spatial_state_interface=='raw':
            from as_jepa_lewm.carrier import RawConditionChartRenderer
            self.renderer_type=RawConditionChartRenderer
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
        self.spatial_basis=spatial_basis
        # No new default buffer/key or tensor operation in legacy checkpoints.
        if spatial_basis=='orthogonal':
            self.register_buffer('spatial_orthogonal_basis',torch.from_numpy(orthogonal_basis()))
        self.spatial_chart=spatial_chart
        self.spatial_state_interface=spatial_state_interface
        if spatial_chart!='none':
            # Reuse the certified residual map. One2128-parameter16→64→16 map
            # acts on every slot; initialization uses its isolated CPU RNG.
            from as_jepa_lewm.carrier import ResidualCarrierChart
            self.carrier_chart_module=ResidualCarrierChart(16,64,spatial_chart,.5)
            if spatial_state_interface=='raw':self.renderer.bind_condition_chart(self.carrier_chart_module)

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

    def transform_spatial_basis(self,state):
        if self.spatial_basis=='identity':return state
        # Fixed diagnostic basis after the existing normalization. FP64
        # accumulation avoids adding a TF32-dependent projection discrepancy;
        # the returned state still has its original dtype (normally FP32).
        with torch.autocast(device_type=state.device.type,enabled=False):
            return (state.double()@self.spatial_orthogonal_basis.double()).to(state.dtype)

    def transform_spatial_chart(self,state):
        if self.spatial_chart=='none':return state
        slots=state.reshape(*state.shape[:-1],self.state_slots,self.state_width)
        return self.carrier_chart_module(slots).reshape_as(state)

    def encode(self,info):
        if not self._spatial_has_statistics:raise RuntimeError('Prepare train-only statistics before encode or training')
        info['emb']=(self.raw_carrier(info['pixels'])-self.spatial_mean)/self.spatial_scale
        info['emb']=self.transform_spatial_basis(info['emb'])
        if self.spatial_state_interface=='chart':info['emb']=self.transform_spatial_chart(info['emb'])
        if 'action' in info:info['act_emb']=self.action_encoder(info['action'])
        return info

    def predict(self,emb,act_emb):
        if self.spatial_state_interface=='chart':return super().predict(emb,act_emb)
        chart_prediction=super().predict(self.transform_spatial_chart(emb),act_emb)
        slots=chart_prediction.reshape(*chart_prediction.shape[:-1],self.state_slots,self.state_width)
        return self.carrier_chart_module.differentiable_inverse(slots).reshape_as(chart_prediction)

    def audit_expected_gradients(self):
        result=dict(encoder='frozen',pool='frozen',predictor='active',renderer='active',action_encoder='optional')
        if self.spatial_chart!='none':
            result['carrier_chart_module']='frozen' if self.spatial_chart=='identity' else 'active'
        return result

    def spatial_parameter_counts(self):
        result=dict(total=sum(p.numel() for p in self.parameters()),
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
            frozen=sum(p.numel() for p in self.parameters() if not p.requires_grad),
            fixed_projection_elements=self.spatial_projection.numel(),
            fixed_projection_bytes=self.spatial_projection.numel()*self.spatial_projection.element_size(),
            fixed_statistics_elements=self.spatial_mean.numel()+self.spatial_scale.numel(),
            state_slots=self.state_slots,state_width=self.state_width,state_dim=self.state_dim)
        if self.spatial_chart!='none':
            result.update(chart=sum(p.numel() for p in self.carrier_chart_module.parameters()),
                chart_trainable=sum(p.numel() for p in self.carrier_chart_module.parameters() if p.requires_grad))
        return result
