"""Episode-disjoint, pixel/action-only windows for public HDF5 control datasets."""
import json
import errno
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def make_training_dataset(cfg):
    if cfg.get("dataset_kind") == "query_rgb_paths":
        from as_jepa_tools.query_rgb_paths import QueryRGBPaths
        return QueryRGBPaths(cfg)
    if cfg.get("dataset_kind") == "goal_rgb_paths":
        from as_jepa_tools.goal_rgb_paths import GoalRGBPaths
        return GoalRGBPaths(cfg)
    if cfg.get("dataset_kind") == "native_rgb_pairs":
        from as_jepa_tools.native_rgb_pairs import NativeRGBPairs
        return NativeRGBPairs(cfg)
    if cfg.get("dataset_kind") == "attainable_pairs":
        from as_jepa_tools.attainable_pairs import AttainablePairs
        return AttainablePairs(cfg)
    if cfg.get("dataset_kind") == "density_transitions":
        from as_jepa_tools.predictive_density_data import DensityTransitions
        return DensityTransitions(cfg)
    if cfg.get("dataset_kind") == "kernel_transitions":
        from as_jepa_tools.predictive_kernel_data import KernelTransitions
        return KernelTransitions(cfg)
    if cfg.get("dataset_kind") == "trajectory_pairs":
        from as_jepa_tools.trajectory_pairs import TrajectoryPairs
        return TrajectoryPairs(cfg)
    if cfg.get("dataset_kind") == "benchmark_hdf5":
        if cfg.get("benchmark_anchor_rule") == "past_gap":
            from as_jepa_tools.benchmark_gap import make_benchmark_dataset
            dataset=make_benchmark_dataset(cfg)
        else:
            dataset=BenchmarkWindows(cfg["data"],window=cfg["model"]["window"],seed=cfg["seed"])
    else:
        from as_jepa_tools.data import FullPushTWindows,PushTWindows
        dataset=(FullPushTWindows if cfg.get("all_phase_data",False) else PushTWindows)(cfg["data"],window=cfg["model"]["window"],gap=cfg.get("gap",8),seed=cfg["seed"])
    if cfg.get("goal_query_reference",False):
        from as_jepa_tools.goal_query_data import QueryReferenceWindows
        dataset=QueryReferenceWindows(dataset,seed=cfg['seed'],mode=cfg.get('goal_query_reference_mode','global'))
    return dataset


class BenchmarkWindows(Dataset):
    def __init__(self,root,window=8,seed=0,split="train"):
        self.root=Path(root);self.card=json.loads((self.root/"dataset_card.json").read_text())
        self.window=window;self.seed=seed;self.stride=self.card["frameskip"]
        self.offsets=np.load(self.root/"offsets.npy");self.lengths=np.load(self.root/"lengths.npy")
        self.episode_ids=np.load(self.root/"episode_ids.npy");selected=np.load(self.root/(split+"_episodes.npy"))
        self.episodes=np.flatnonzero(np.isin(self.episode_ids,selected)&(self.lengths>(window-1)*self.stride))
        self.counts=self.lengths[self.episodes]-(window-1)*self.stride
        self.ends=np.cumsum(self.counts);self.natural_length=int(self.ends[-1]);self.encode_epoch_index=True
        self.mean=np.asarray(self.card["action_mean"],dtype=np.float32);self.std=np.asarray(self.card["action_std"],dtype=np.float32)
        self.file=None;self.actions=None;self.pixels=None
    def __len__(self):return self.natural_length
    def locate(self,index):
        index=int(index)%self.natural_length;i=int(np.searchsorted(self.ends,index,side="right"))
        return int(self.episodes[i]),index-(int(self.ends[i-1]) if i else 0)
    def open(self):
        if self.file is None:
            import h5py,hdf5plugin
            self.file=h5py.File(self.card["h5_path"],"r",rdcc_nbytes=32*1024*1024)
            self.actions=np.asarray(self.file["action"],dtype=np.float32)
            # Keep the VDS Dataset alive: releasing it after each window also
            # releases its source-file/chunk caches on remote filesystems.
            self.pixels=self.file["pixels"]
    def __getitem__(self,index):
        for attempt in range(3):
            try:return self._item(index)
            except OSError as error:
                if error.errno!=errno.EIO or attempt==2:raise
                # Retry the SAME read-only sample with fresh VDS/source handles.
                # Never skip a row,replace pixels or change a random stream.
                if self.file is not None:self.file.close()
                self.file=None;self.actions=None;self.pixels=None
                time.sleep(1+attempt)
    def _item(self,index):
        self.open();ep,start=self.locate(index);offset=int(self.offsets[ep]);n=int(self.lengths[ep]);w=self.window
        rows=offset+start+np.arange(w)*self.stride
        # Read the small enclosing interval before selecting rows in NumPy.
        # HDF5 fancy selection is expensive for the supplied 100-frame chunks.
        # This preserves every selected byte, including the final valid window.
        raw=np.asarray(self.pixels[int(rows[0]):int(rows[-1])+1])[::self.stride]
        pixels=torch.from_numpy(raw.copy())
        if pixels.shape[-1] in (3,4):pixels=pixels[...,:3].permute(0,3,1,2)
        if pixels.shape[1]!=3:raise ValueError("Expected RGB pixels")
        if pixels.shape[-2:]!=(224,224):pixels=F.interpolate(pixels.float(),size=(224,224),mode="bilinear",align_corners=False,antialias=True).round().clamp(0,255).byte()
        render=F.interpolate(pixels.float(),size=(64,64),mode="area").round().byte()
        actions=np.zeros((w,self.stride,len(self.mean)),dtype=np.float32)
        for t,row in enumerate(rows[:-1]):
            block=self.actions[row:row+self.stride]
            if not np.isfinite(block).all():raise ValueError("Nonfinite action in a valid transition")
            actions[t]=(block-self.mean)/self.std
        return dict(pixels=pixels,render=render,anchor=render[0],anchor224=pixels[0],action=torch.from_numpy(actions.reshape(w,-1)),episode=int(self.episode_ids[ep]),start=start,rows=torch.from_numpy(rows.copy()),anchor_row=int(rows[0]),anchor_fallback=False)
