"""Thin interfaces around pinned public models and verbatim native loss functions.

No baseline loss is substituted by the candidate's loss. Importing function ASTs
avoids executing unrelated Hydra/WandB/cluster entry points in public train.py.
"""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from einops import rearrange

SOURCES = Path('/efs/prescottsun/SPT-JEPA/sources/metaworld-baselines-20261107')
METHODS = ['lewm', 'lewm-idm', 'h-jepa', 'smwm', 'ac-mtm', 'lpwm']


def load_file(name, path):
    name = 'mw_public_' + name
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def native_functions(path, names, extra=None):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    ast.fix_missing_locations(code)
    namespace = dict(torch=torch, F=F, rearrange=rearrange)
    namespace.update(extra or {})
    exec(compile(code, str(path), 'exec'), namespace)
    return namespace


class PublicBaseline(nn.Module):
    def __init__(self, method, sources=str(SOURCES)):
        super().__init__()
        assert method in METHODS
        self.method = method
        self.sources = Path(sources)
        self.regularizers = nn.ModuleDict()
        self.history = 7 if method in ['lewm-idm', 'h-jepa'] else (1 if method == 'smwm' else 3)
        self.frames = self.history + 1
        self.cfg = None
        self._construct()

    def _construct(self):
        method = self.method
        if method in ['lewm-idm', 'h-jepa']:
            # Its initializer owns the eval resolver; import it before composing
            # H-JEPA configs, including when this is the first model in a process.
            import stable_pretraining
            folder = self.sources/'hjepa/h_jepa'
            sys.path.insert(0, str(folder))
            from hydra import compose, initialize_config_dir
            if not OmegaConf.has_resolver('eval'):
                OmegaConf.register_new_resolver('eval', eval)
            with initialize_config_dir(version_base=None, config_dir=str(folder/'config/train')):
                cfg = compose(config_name='droid_hjepa_l2' if method == 'h-jepa' else 'droid_lewm')
            cfg.img_size = 224
            cfg.level1.encoder.image_size = 224
            cfg.level1.wm.action_dim = 20
            cfg.data.dataset.keys_to_load = ['pixels', 'action']
            from hjepa_utils import create_world_model, hjepa_forward
            self.core, losses = create_world_model(cfg)
            self.regularizers.update(losses)
            self.cfg = cfg
            self.native_loss = hjepa_forward
            return

        folder = self.sources/({'lewm':'lewm', 'smwm':'smwm/planning', 'ac-mtm':'acmtm', 'lpwm':'lpwm/lpwm_swm'}[method])
        mod = load_file(method.replace('-', '_')+'_module', folder/'module.py')
        jepa = load_file(method.replace('-', '_')+'_model', folder/('model.py' if method == 'lpwm' else 'jepa.py'))
        from stable_pretraining.backbone.utils import vit_hf
        encoder = vit_hf(size='tiny', patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
        predictor_type = mod.Predictor if method == 'lpwm' else mod.ARPredictor
        predictor = predictor_type(num_frames=self.history, input_dim=192, hidden_dim=192, output_dim=192,
            depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=.1, emb_dropout=0.)
        mlp_extra = dict(link_function_type='ReLU') if method == 'lpwm' else {}
        parts = dict(encoder=encoder, predictor=predictor, action_encoder=mod.Embedder(input_dim=20, emb_dim=192),
            projector=mod.MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=nn.BatchNorm1d, **mlp_extra),
            pred_proj=mod.MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=nn.BatchNorm1d, **mlp_extra))
        if method == 'smwm':
            parts['inverse_model'] = mod.InverseModel(embed_dim=192, action_dim=20, hidden_dim=256)
        elif method == 'ac-mtm':
            parts['inverse_predictor'] = mod.InverseDynamics(latent_dim=192, action_dim=20, hidden_dim=512, depth=2, dropout=0.)
        elif method == 'lpwm':
            parts['history_size'] = self.history
        self.core = (jepa.LpWM if method == 'lpwm' else jepa.JEPA)(**parts)
        if method == 'lewm':
            self.cfg = OmegaConf.create(dict(history_size=3, num_preds=1, loss=dict(sigreg=dict(weight=.09))))
            self.regularizers['sigreg'] = mod.SIGReg(knots=17, num_proj=1024)
            self.native_loss = native_functions(folder/'train.py', ['lejepa_forward'])['lejepa_forward']
        elif method == 'smwm':
            self.cfg = OmegaConf.create(dict(wm=dict(history_size=1), loss=dict(sigreg=dict(weight=0.), inverse=dict(weight=1.))))
            self.native_loss = native_functions(folder/'train.py', ['forward_step'])['forward_step']
        elif method == 'ac-mtm':
            self.cfg = OmegaConf.create(dict(wm=dict(history_size=3, num_preds=1), loss=dict(sigreg=dict(weight=0.),
                masked=dict(mask_mode='both', inverse_weight=.30, inverse_warmup_epochs=0,
                    inverse_loss_type='action_nce', action_nce_temperature=.1))))
            names = ['_representation_diagnostics', '_inverse_margin_diagnostics', 'action_discrimination_loss', 'masked_transition_forward']
            self.native_loss = native_functions(folder/'train.py', names)['masked_transition_forward']
        else:
            loss = load_file('lpwm_loss', folder/'loss.py')
            self.regularizers['rdmreg'] = loss.RDMReg(p=1., mu=0., link_function_type='ReLU',
                loss_func_type='sliced_wasserstein', num_slices=1024, matching_mode='b_t_d', target_dist_rms_norm=False)
            self.regularizers['temporal_jaccard'] = loss.TemporalJaccardLoss(eps=1e-8)
            self.cfg = OmegaConf.create(dict(wm=dict(history_size=3, num_preds=1), loss=dict(
                rdmreg=dict(weight=10.), temporal_jaccard=dict(enabled=True, weight=.1))))
            # Exactly the public Cube transfer recipe; all MetaWorld tasks share it.
            metrics = load_file('lpwm_metrics', folder/'metrics.py')
            self.native_loss = native_functions(folder/'train.py', ['lpjepa_forward'], {
                key:getattr(metrics,key) for key in ['l1_sparsity_metric','l0_sparsity_metric','avg_per_dim_var','off_diag_cov_sum']})['lpjepa_forward']

    def normalize(self, pixels):
        if self.method == 'smwm':
            return pixels  # public SMWM.encode performs its own ImageNet normalization
        value = pixels.float()/255. if pixels.dtype == torch.uint8 else pixels
        shape = [1]*(value.ndim-3)+[3,1,1]
        mean = value.new_tensor([.485,.456,.406]).view(shape)
        std = value.new_tensor([.229,.224,.225]).view(shape)
        return (value-mean)/std

    def forward(self, batch, epoch=0):
        assert set(batch) == {'pixels', 'action'}
        assert batch['pixels'].shape[1] == self.frames
        assert batch['action'].shape[1:] == (self.frames,20)
        proxy = SimpleNamespace(model=self.core, current_epoch=epoch, log_dict=lambda *a, **kw: None,
            **dict(self.regularizers.items()))
        if self.method in ['lewm-idm', 'h-jepa']:
            data = dict(pixels_level1=batch['pixels'], action_level1=batch['action'])
            def normalize(value):
                value['pixels_level1'] = self.normalize(value['pixels_level1'])
            result = self.native_loss(proxy, data, 'fit', self.cfg, normalize_batch=normalize)
        else:
            data = dict(pixels=self.normalize(batch['pixels']), action=batch['action'])
            result = self.native_loss(proxy, data, 'fit', self.cfg)
        metrics = {k:v.detach().float() for k,v in result.items() if torch.is_tensor(v) and v.numel()==1}
        return result['loss'], metrics

    def encode(self, info):
        if self.method in ['lewm-idm', 'h-jepa']:
            encoded = self.core.get_level(1).encode(dict(info))
            info['emb'] = encoded['embed_0']
            if 'action_0' in encoded:
                info['act_emb'] = encoded['action_0']
            return info
        return self.core.encode(info)

    def predict(self, emb, action):
        core = self.core.get_level(1) if self.method in ['lewm-idm', 'h-jepa'] else self.core
        return core.predict(emb, action)

    def embed_actions(self, action):
        if self.method in ['lewm-idm', 'h-jepa']:
            return self.core.get_level(1).action_embed(action)
        return self.core.action_encoder(action)

    def rollout(self, info, actions):
        """Common MPC interface: factual past blocks precede imagined actions.

        The predictor's native maximum history is retained; initial physical
        context is the shared two frames. No inverse head is used for control.
        """
        b,s,h,a = actions.shape
        pixels = info['pixels'][:,0]
        if 'emb' not in info:
            initial = self.encode(dict(pixels=pixels))['emb']
            info['emb'] = initial[:,None].expand(b,s,*initial.shape[1:])
        emb = info['emb'].flatten(0,1)
        context = emb.size(1)
        past = info.get('action_history')
        if context > 1:
            assert past is not None and past.shape[2:] == (context-1,20)
            complete_actions = torch.cat([past, actions], dim=2)
        else:
            complete_actions = actions
        act = self.embed_actions(complete_actions.flatten(0,1))
        for step in range(h):
            end = context+step
            start = max(0,end-self.history)
            next_emb = self.predict(emb[:,start:end], act[:,start:end])[:,-1:]
            emb = torch.cat([emb,next_emb],dim=1)
        info['predicted_emb'] = emb.unflatten(0,(b,s))
        return info

    def optimizer_spec(self):
        if self.method in ['lewm-idm', 'h-jepa']:
            groups = [dict(params=list(self.core.get_level(i).parameters()), lr=5e-4, name=f'level{i}')
                for i in range(1,self.core.num_levels+1)]
            return groups, dict(lr=5e-4, weight_decay=1e-4), .1
        lr = 1e-4 if self.method == 'smwm' else 5e-5
        return self.parameters(), dict(lr=lr,weight_decay=1e-3), .01

    def gradient_groups(self):
        if self.method in ['lewm-idm','h-jepa']:
            return [f'core.jepas.{i}.{part}' for i in range(self.core.num_levels)
                for part in ['encoder','predictor','idm_loss_fn']]
        parts = ['encoder','predictor','action_encoder','projector','pred_proj']
        if self.method == 'smwm':parts.append('inverse_model')
        if self.method == 'ac-mtm':parts.append('inverse_predictor')
        return ['core.'+part for part in parts]
