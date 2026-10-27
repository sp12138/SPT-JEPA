from vforge.models.as_jepa.constants import C,PATCH

def pack(frames,patch=PATCH):
    n,c,h,w=frames.shape
    return frames.reshape(n,c,h//patch,patch,w//patch,patch).permute(0,3,5,1,2,4).reshape(n,c*patch*patch,1,h//patch,w//patch)

def unpack(packed,patch=PATCH,c=C):
    n,_,_,gh,gw=packed.shape
    return packed.reshape(n,patch,patch,c,gh,gw).permute(0,3,4,1,5,2).reshape(n,c,gh*patch,gw*patch)
