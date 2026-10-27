"""Persistent, restartable PushT export. Shared HDF5 input is opened read-only."""
import argparse, concurrent.futures, hashlib, json, os, time
from pathlib import Path
import h5py
import numpy as np


def export_part(arguments):
    source, output, first, last = arguments
    import torch
    torch.set_num_threads(2)
    out = Path(output)
    frames = np.load(out / "pixels224.npy", mmap_mode="r+")
    small = np.load(out / "pixels64.npy", mmap_mode="r+")
    offsets = np.load(out / "offsets.npy")
    lengths = np.load(out / "lengths.npy")
    raw_offsets = np.load(out / "raw_offsets.npy")
    raw_lengths = np.load(out / "raw_lengths.npy")
    marker = out / f"part_{first:05d}_{last:05d}.json"
    if marker.exists(): return json.loads(marker.read_text())
    started = time.time()
    with h5py.File(source, "r") as f:
        for ep in range(first, last):
            lo, n = int(offsets[ep]), int(lengths[ep])
            raw, count = int(raw_offsets[ep]), int(raw_lengths[ep])
            image = f["pixels"][raw:raw + count:5]
            assert len(image) == n
            frames[lo:lo+n] = image
            tensor = torch.from_numpy(image).permute(0,3,1,2).float()
            render = torch.nn.functional.interpolate(tensor, size=(64,64), mode="area").round().clamp_(0,255).byte()
            small[lo:lo+n] = render.permute(0,2,3,1).numpy()
    frames.flush(); small.flush()
    record = dict(first=first,last=last,seconds=time.time()-started,frames=int(lengths[first:last].sum()))
    marker.write_text(json.dumps(record))
    return record


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--source",required=True); parser.add_argument("--output",required=True); parser.add_argument("--workers",type=int,default=12)
    args=parser.parse_args(); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    if (output/"ready.json").exists():print((output/"ready.json").read_text());return
    with h5py.File(args.source,"r") as f:
        raw_lengths=f["ep_len"][:].astype(np.int64); raw_offsets=f["ep_offset"][:].astype(np.int64)
        # ep_offset may carry a terminal sentinel; use one entry per episode.
        raw_offsets=raw_offsets[:len(raw_lengths)]
        lengths=(raw_lengths+4)//5; offsets=np.r_[0,np.cumsum(lengths[:-1])].astype(np.int64)
        rows=np.concatenate([np.arange(int(o),int(o+n),5,dtype=np.int64) for o,n in zip(raw_offsets,raw_lengths)])
        for key,value in dict(raw_lengths=raw_lengths,raw_offsets=raw_offsets,lengths=lengths,offsets=offsets,raw_rows=rows).items():np.save(output/f"{key}.npy",value)
        for name,shape in [("pixels224",(len(rows),224,224,3)),("pixels64",(len(rows),64,64,3))]:
            path=output/f"{name}.npy"
            if not path.exists():
                array=np.lib.format.open_memmap(path,mode="w+",dtype=np.uint8,shape=shape);array.flush();del array
        action=np.nan_to_num(f["action"][:],nan=0.0); state=f["state"][:]
        good=np.isfinite(f["action"][:]).all(1)
        mean=action[good].mean(0); std=action[good].std(0).clip(1e-6)
        np.save(output/"action_raw.npy",action)
        np.save(output/"states.npy",state[rows])
        actions=np.stack([action[np.minimum(rows+step,len(action)-1)] for step in range(5)],axis=1)
        # Zero padded blocks crossing episode boundaries; those blocks are not used as transitions.
        normalized=((actions-mean)/std).reshape(len(rows),10).astype(np.float32)
        np.save(output/"actions.npy",normalized)
        rng=np.random.default_rng(0); test=np.sort(rng.permutation(len(lengths))[:round(.05*len(lengths))]); np.save(output/"test_episodes.npy",test)
        metadata=dict(source=str(Path(args.source).resolve()),source_bytes=Path(args.source).stat().st_size,episodes=len(lengths),raw_rows=len(action),kept_frames=len(rows),frameskip=5,phase=0,window_start_stride_raw=5,split_seed=0,test_episodes=len(test),action_mean=mean.tolist(),action_std=std.tolist(),note="Insight campaign uses one frameskip-aligned phase and every kept-frame window start, rather than the original spec's all five raw-frame phases. Encoder remains 224px; rendering is uint8-rounded 64px area resize.",input_read_only=True)
        (output/"dataset_card.json").write_text(json.dumps(metadata,indent=2))
    boundaries=list(range(0,len(lengths),256))
    work=[(args.source,str(output),first,min(first+256,len(lengths))) for first in boundaries]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i,result in enumerate(executor.map(export_part,work)):
            print(json.dumps(dict(completed=i+1,total=len(work),**result)),flush=True)
    (output/"ready.json").write_text(json.dumps(dict(ready=True,completed_at=time.time(),**metadata),indent=2))

if __name__=="__main__":main()
