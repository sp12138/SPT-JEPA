"""Inspect the declared16-step sampler on retained factual audit controls."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from .train import setup
from .composition import sample,mixture,coefficients


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=read(BASE/f'runs/metaworld_endpoint_density_20261109/training/{a.name}/config.json')
    _,parent,head,_,_=setup(cfg);head.eval().requires_grad_(False)
    weights=Path(cfg['output'])/'head-step0002000/weights.pt';assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    head.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True));before=(tensor_sha(parent),tensor_sha(head))
    bank=BASE/'runs/metaworld_realized_bank_20261109/push-v3-seed6042.npz'
    assert sha(bank)==read(bank.with_suffix('.json'))['bank_sha256'];data=np.load(bank)
    with torch.inference_mode():
        past=torch.from_numpy(data['warm_history'].copy()).permute(0,3,1,2)[None].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden,current=parent.prepare(dict(history_pixels=past,history_actions=torch.zeros(1,7,4,device='cuda'),history_valid=torch.ones(1,8,device='cuda',dtype=torch.bool)))
            hidden=hidden.expand(4,-1,-1,-1).clone();future=current.expand(4,-1,-1,-1)
            commands=torch.from_numpy(data['controls'][:4].copy()).cuda()
            for t in range(60):hidden,future=parent.predict(hidden,future,commands[:,t])
        anchor=(current*2-1).expand(4,-1,-1,-1)
        generated=sample(head,hidden,anchor,20261109,particles=2)
        duplicate=sample(head,hidden,anchor,20261109,particles=2)
        assert torch.equal(generated,duplicate) and torch.isfinite(generated).all()
        assert generated.min()>=-1 and generated.max()<=1 and generated.std()>.01
        const=torch.full((2,4),3.2,device='cuda');assert torch.allclose(mixture(const),const[0].double(),atol=1e-6)
        canvas=Image.new('RGB',(5*224,4*245),'white');draw=ImageDraw.Draw(canvas)
        def picture(x):return Image.fromarray(((x+1)*127.5).clamp(0,255).byte().permute(1,2,0).cpu().numpy()).resize((224,224))
        actual=torch.from_numpy(data['actual60'][:4].copy()).permute(0,3,1,2).cuda().float()/127.5-1
        for i in range(4):
            frames=[current[0]*2-1,future[i]*2-1,generated[i],generated[4+i],actual[i]]
            for j,(label,frame) in enumerate(zip(['Observed start','Old residual forecast','Reverse-chain sample1','Reverse-chain sample2','Actually realized'],frames)):
                canvas.paste(picture(frame),(j*224,i*245));draw.text((j*224+3,i*245+225),label,fill='black')
        a.output.mkdir(parents=True,exist_ok=False);figure=a.output/'sampler-physical-comparison.png';canvas.save(figure)
        error=((generated+1)/2-((actual+1)/2).repeat(2,1,1,1)).square().mean((1,2,3))
    assert before==(tensor_sha(parent),tensor_sha(head))
    save(a.output/'AUDIT.json',dict(passed=True,utc=now(),model=a.name,figure=str(figure),figure_sha256=sha(figure),
        deterministic_private_rng_replay_exact=True,finite_quantized_8bit_outputs=True,sample_actual_rgb_mse=error.cpu().tolist(),
        parent_and_head_unchanged=True,model_updates=0,physics_steps=0,candidate_performance=False,training_data_eligible=False,
        scope='First4ownGaussian/zeroactualauditpaths,notexpertororacleproposals. Sampleruses16reverseGaussiansteps+endpoint, '
            'samechainasdeclaredgoalbound. Goalnotaninputtosampler. Theseprotectedimages/controlsNEVERforSGD/proposals.'))
