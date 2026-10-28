"""Export decoded pinned Lance frames once and index every raw window start."""
import argparse, concurrent.futures, hashlib, io, json, time
from pathlib import Path
import numpy as np

def export_chunk(args):
    source,output,first,last=args
    import lance,torch
    from PIL import Image
    torch.set_num_threads(2)
    out=Path(output);marker=out/"parts"/f"{first:08d}.json"
    if marker.exists():return
    dataset=lance.dataset(source)
    blobs=dataset.take(list(range(first,last)),columns=["pixels"])["pixels"].to_pylist()
    image=np.stack([np.asarray(Image.open(io.BytesIO(x)).convert("RGB")) for x in blobs])
    assert image.shape[1:]==(224,224,3)
    large=np.load(out/"pixels224.npy",mmap_mode="r+");small=np.load(out/"pixels64.npy",mmap_mode="r+")
    large[first:last]=image
    tensor=torch.from_numpy(image).permute(0,3,1,2).float()
    small[first:last]=torch.nn.functional.interpolate(tensor,size=(64,64),mode="area").round().byte().permute(0,2,3,1).numpy()
    large.flush();small.flush()
    marker.write_text(json.dumps(dict(first=first,last=last,pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())))

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--output",required=True);p.add_argument("--workers",type=int,default=12);a=p.parse_args()
    import lance
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/"ready.json").exists():return
    (out/"parts").mkdir(exist_ok=True)
    data=lance.dataset(a.source)
    table=data.to_table(columns=["episode_idx","step_idx","action","state"])
    episodes=table["episode_idx"].to_numpy();steps=table["step_idx"].to_numpy()
    offsets=np.flatnonzero(np.r_[True,episodes[1:]!=episodes[:-1]])
    lengths=np.diff(np.r_[offsets,len(episodes)])
    assert len(offsets)==18685 and len(episodes)==2336736
    assert np.array_equal(episodes[offsets],np.arange(len(offsets)))
    assert all(np.array_equal(steps[o:o+n],np.arange(n)) for o,n in zip(offsets,lengths))
    action=np.array(table["action"].to_pylist(),dtype=np.float32)
    raw_state=np.array(table["state"].to_pylist(),dtype=np.float32)
    valid=np.isfinite(action).all(1);mean=action[valid].mean(0);std=action[valid].std(0).clip(1e-6)
    action=np.nan_to_num(action,nan=0.)
    rows=np.arange(len(episodes))
    limits=np.repeat(offsets+lengths-1,lengths)
    blocks=np.stack([action[np.minimum(rows+k,limits)] for k in range(5)],1)
    normalized=((blocks-mean)/std).reshape(len(rows),10).astype(np.float32)
    test=np.sort(np.random.default_rng(0).permutation(len(lengths))[:round(.05*len(lengths))])
    np.save(out/"action_raw.npy",action)
    arrays=dict(offsets=offsets,lengths=lengths,raw_rows=rows,states=raw_state,actions=normalized,test_episodes=test,episode_indices=episodes,step_indices=steps)
    for name,array in arrays.items():
        path=out/(name+".npy")
        if not path.exists():np.save(path,array)
    for name,side in [("pixels224",224),("pixels64",64)]:
        path=out/(name+".npy")
        if not path.exists():np.lib.format.open_memmap(path,mode="w+",dtype=np.uint8,shape=(len(rows),side,side,3)).flush()
    card=dict(source=a.source,revision="ea321e392348e3c65a18ab0d685f00e57be2c3e0",format="lance JPEG decoded RGB",frameskip=5,phases=5,episodes=len(lengths),raw_rows=len(rows),split_seed=0,test_episodes=len(test),action_mean=mean.tolist(),action_std=std.tolist(),normalization="finite rows population mean/std,NaN actions zero; padded terminal blocks excluded from transitions")
    (out/"dataset_card.json").write_text(json.dumps(card,indent=2))
    tasks=[(a.source,str(out),first,min(first+512,len(rows))) for first in range(0,len(rows),512)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,_ in enumerate(pool.map(export_chunk,tasks)):
            if i%32==0:(out/"progress.json").write_text(json.dumps(dict(completed=i+1,total=len(tasks),updated_at=time.time())))
    # Phase0 evaluation uses views into the same decoded raw bank; no duplicate image bank.
    view=out/"phase0";view.mkdir(exist_ok=True)
    view_rows=np.concatenate([np.arange(o,o+n,5) for o,n in zip(offsets,lengths)])
    view_lengths=(lengths+4)//5;view_offsets=np.r_[0,np.cumsum(view_lengths[:-1])]
    for name,array in dict(offsets=view_offsets,lengths=view_lengths,raw_rows=view_rows,states=raw_state[view_rows],actions=normalized[view_rows],test_episodes=test,pixel_row_indices=view_rows).items():np.save(view/(name+".npy"),array)
    view_card={**card,"phases":1,"phase":0,"raw_storage":str(out),"kept_frames":len(view_rows)}
    (view/"action_raw.npy").symlink_to(out/"action_raw.npy") if not (view/"action_raw.npy").exists() else None
    (view/"dataset_card.json").write_text(json.dumps(view_card,indent=2))
    validation=[]
    from PIL import Image
    import torch
    pixels=np.load(out/"pixels224.npy",mmap_mode="r");small=np.load(out/"pixels64.npy",mmap_mode="r")
    for index in np.random.default_rng(20261028).choice(len(rows),1000,replace=False):
        blob=data.take([int(index)],columns=["pixels"])["pixels"][0].as_py()
        original=np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
        assert np.array_equal(original,pixels[index])
        target=torch.nn.functional.interpolate(torch.from_numpy(original.copy()).permute(2,0,1)[None].float(),size=(64,64),mode="area").round().byte()[0].permute(1,2,0).numpy()
        assert np.array_equal(target,small[index])
        validation.append(int(index))
    report=dict(ready=True,completed_at=time.time(),random_pixel_checks=1000,validation_indices_sha256=hashlib.sha256(np.array(validation).tobytes()).hexdigest(),split_overlap=0,**card)
    (out/"ready.json").write_text(json.dumps(report,indent=2));(view/"ready.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=="__main__":main()
