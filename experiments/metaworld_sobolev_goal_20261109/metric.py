"""Fixed spatial goal geometry; no learned targets or task-specific parameters.

For RGB difference d, use sum |DCT(d)[k,l]|^2/(1+k^2+l^2).
The positive spectrum makes this an image metric. Reflecting boundaries avoid
the left/right wraparound of a periodic Fourier kernel. Raw pixel MSE gives
equal cost to disjoint translations of a fixed object; this kernel couples
separated pixels. Whether that helps real control is an experimental question.
"""
import functools
import math
import torch


@functools.lru_cache(maxsize=16)
def basis(size,device):
    n=torch.arange(size,dtype=torch.float64)
    k=torch.arange(size,dtype=torch.float64)[:,None]
    result=torch.cos(math.pi*(n[None]+.5)*k/size)*math.sqrt(2/size)
    result[0]*=1/math.sqrt(2)
    return result.float().to(device)


def cost(predicted,goal):
    difference=predicted.float()-goal.float();h,w=difference.shape[-2:]
    transformed=basis(h,str(difference.device))@difference@basis(w,str(difference.device)).T
    y=torch.arange(h,device=difference.device,dtype=torch.float32)[:,None]
    x=torch.arange(w,device=difference.device,dtype=torch.float32)[None,:]
    weight=1/(1+x.square()+y.square())
    return (transformed.square()*weight).mean((1,2,3))
