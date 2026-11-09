"""Observed-frame geometric codecs plus an executed-input latent world.

Feature transport follows the principle in Kulkarni et al., NeurIPS2019,
https://arxiv.org/abs/1906.11883 . This is an adaptation, not a reproduction
or a novelty claim. No keypoint/segmentation/state/action labels are used.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from experiments.metaworld_joint_flow_20261109.model import JointFlow,random_terms


class GeometryWorld(JointFlow):
    def __init__(self,parent,renderer,mode):
        super().__init__(parent,renderer,'highnoise')
        assert mode in ['dense','sparse','transport'];self.codec_mode=mode
        self.renderer.requires_grad_(False);self.prior_scale.requires_grad_(False);self.posterior_scale.requires_grad_(False)
        self.keyvision=copy.deepcopy(parent.vision);self.keyhead=nn.Conv2d(96,16,1)
        self.future_features=nn.Sequential(nn.Conv2d(128,192,1),nn.GroupNorm(8,192),nn.SiLU(),nn.Conv2d(192,192,3,padding=1))
        self.codec_decoder=nn.Sequential(nn.Conv2d(96,128,3,padding=1),nn.GroupNorm(8,128),nn.SiLU(),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(32,16,3,padding=1),nn.GroupNorm(8,16),nn.SiLU(),
            nn.Conv2d(16,3,1),nn.Tanh())
        nn.init.normal_(self.keyhead.weight,std=.1);nn.init.zeros_(self.keyhead.bias)
        y,x=torch.meshgrid(torch.linspace(-1,1,16),torch.linspace(-1,1,16),indexing='ij')
        self.register_buffer('grid',torch.stack([x,y],-1))
        if mode=='dense':self.keyvision.requires_grad_(False);self.keyhead.requires_grad_(False)

    def points(self,appearance,key_features,detach_head=False):
        logits=F.conv2d(key_features,self.keyhead.weight.detach(),self.keyhead.bias.detach()) if detach_head else self.keyhead(key_features)
        with torch.autocast(device_type=appearance.device.type,enabled=False):
            probability=logits.float().flatten(2).softmax(-1)
            centers=torch.einsum('bkp,pd->bkd',probability,self.grid.float().flatten(0,1))
            descriptors=F.grid_sample(appearance.float(),centers[:,:,None,:],mode='bilinear',align_corners=True).squeeze(-1).transpose(1,2)
            descriptors=F.normalize(descriptors,dim=-1,eps=1e-8)
            # One feature-grid cell, independent of task/object identity.
            masks=(-(self.grid.float()[None,None]-centers[:,:,None,None,:]).square().sum(-1)/(2*(2/15)**2)).exp()
        return centers,descriptors,masks

    def observed(self,image):
        appearance=self.parent.vision(image);key=self.keyvision(image)
        return appearance,key

    def predicted(self,hidden):return self.future_features(hidden).chunk(2,1)

    def objective(self,batch,random):
        durations=random[0];prior,current,target=self.factual(batch,durations)
        future_appearance,future_key=self.observed((target+1)/2)
        pred_appearance,pred_key=self.predicted(prior)
        if self.codec_mode=='dense':
            code=future_appearance
            predicted=F.normalize(pred_appearance.float(),dim=1,eps=1e-8)
            actual=F.normalize(future_appearance.detach().float(),dim=1,eps=1e-8)
            coordinate=predicted.new_zeros(len(predicted))
            feature=(predicted-actual).square().sum(1).mean((1,2))
        else:
            actual_xy,actual_d,masks=self.points(future_appearance,future_key)
            with torch.no_grad():
                source_appearance,source_key=self.observed((current+1)/2)
                _,_,source_masks=self.points(source_appearance,source_key)
            code=source_appearance.detach() if self.codec_mode=='transport' else torch.zeros_like(future_appearance)
            for k in range(16):
                destination=masks[:,k:k+1]
                if self.codec_mode=='transport':code=(1-source_masks[:,k:k+1])*(1-destination)*code
                else:code=(1-destination)*code
                code=code+destination*future_appearance
            xy,descriptor,_=self.points(pred_appearance,pred_key,detach_head=True)
            coordinate=(xy-actual_xy.detach()).square().mean((1,2))
            feature=(descriptor-actual_d.detach()).square().sum(-1).mean(-1)
        reconstructed=self.codec_decoder(code).float()
        reconstruction=(reconstructed-target).square().mean((1,2,3))
        loss=(reconstruction+coordinate+feature).mean()
        return loss,dict(loss=loss,reconstruction=reconstruction.mean(),future_coordinates=coordinate.mean(),
            future_unit_features=feature.mean(),duration=durations.float().mean())

    def goal_cost(self,hidden,goal):
        predicted_a,predicted_k=self.predicted(hidden)
        goal_a,goal_k=self.observed(goal)
        if self.codec_mode=='dense':
            return (F.normalize(predicted_a.float(),dim=1,eps=1e-8)-F.normalize(goal_a.float(),dim=1,eps=1e-8)).square().sum(1).mean((1,2))
        xy,descriptor,_=self.points(predicted_a,predicted_k)
        target_xy,target_descriptor,_=self.points(goal_a,goal_k)
        return (xy-target_xy).square().mean((1,2))+(descriptor-target_descriptor).square().sum(-1).mean(-1)
