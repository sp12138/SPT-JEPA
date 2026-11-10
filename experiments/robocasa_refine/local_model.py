"""Local observation codes; the original dynamics still couples every slot.

Masking only the readout does not make a ViT token local: its backbone has
already mixed the full image. Here every encoder layer obeys the same region
partition. CLS reads only itself, so it cannot carry distant image content.
No learned parameters or auxiliary supervision are added.
"""
import types

import torch
from torch.nn import functional as F

from experiments.robocasa_compare.spatial_model import SpatialForecast


def regional_attention(self,hidden_states,**kwargs):
    assert kwargs.get("head_mask") is None
    batch,length,_=hidden_states.shape
    assert length==self._spt_local_allow.shape[0]
    shape=(batch,length,self.num_attention_heads,self.attention_head_size)
    query=self.query(hidden_states).reshape(shape).transpose(1,2)
    key=self.key(hidden_states).reshape(shape).transpose(1,2)
    value=self.value(hidden_states).reshape(shape).transpose(1,2)
    output=F.scaled_dot_product_attention(query,key,value,attn_mask=self._spt_local_allow,
        dropout_p=self.dropout_prob if self.training else 0.,scale=self.scaling,is_causal=False)
    return output.transpose(1,2).reshape(batch,length,self.all_head_size),None


class LocalSpatialForecast(SpatialForecast):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        side=self.encoder.config.image_size//self.encoder.config.patch_size
        assert side==16 and self.state_slots==16
        ids=torch.arange(side*side)
        region=(ids//side//4)*4+(ids%side//4)
        allow=torch.zeros(1+side*side,1+side*side,dtype=torch.bool)
        allow[0,0]=True
        allow[1:,0]=True  # Image-independent constant CLS is safe to read.
        allow[1:,1:]=region[:,None]==region[None,:]
        for layer in self.encoder.encoder.layer:
            attention=layer.attention.attention
            assert hasattr(attention,"dropout_prob") and hasattr(attention,"scaling")
            attention.register_buffer("_spt_local_allow",allow.clone(),persistent=False)
            attention.forward=types.MethodType(regional_attention,attention)
