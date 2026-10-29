"""Keep public task semantics; freeze our own disjoint validation and test episode sets."""
import argparse,json,hashlib,subprocess
from pathlib import Path
from dataclasses import replace
import numpy as np
import clear_lewm.manifests as manifests
import clear_lewm.tasks as tasks
from clear_lewm.protocols import get_protocol

def main():
    p=argparse.ArgumentParser();p.add_argument("task",choices=["cube","tworoom"]);a=p.parse_args()
    b=Path("/efs/prescottsun/SPT-JEPA");root=b/"data/hardtasks-v1"/(a.task+"-metadata");card=json.loads((root/"dataset_card.json").read_text())
    # HDF5 fancy indexing requires sorted unique rows. Preserve the public
    # sampler's exact order and duplicates through an inverse index.
    def read_rows(dataset, key, rows):
        unique,inverse=np.unique(rows,return_inverse=True)
        return np.asarray(dataset[key][unique])[inverse]
    tasks._read_pair=lambda dataset,key,starts,goals:(read_rows(dataset,key,starts),read_rows(dataset,key,goals))
    train=np.load(root/"train_episodes.npy");out=root/"manifests";out.mkdir(exist_ok=False)
    for split,count in [("validation",50),("test",100)]:
        allowed=np.load(root/(split+"_episodes.npy"))
        # The external generator owns geometry, source-window validity and pair balancing.
        # Only its split provider is bound to the already frozen campaign episode split.
        manifests.split_episode_ids=lambda episodes,fraction,seed:(train,allowed)
        for mode in ["moderate","strict"]:
            spec=replace(get_protocol(mode),split="heldout",heldout_fraction=.05,description=get_protocol(mode).description+" SPT-JEPA fixed disjoint episode split; not a canonical CLEAR manifest.")
            m=manifests.generate_manifest(card["h5_path"],a.task,spec,count,42,split="heldout")
            assert set(x["episode_id"] for x in m["pairs"]).issubset(set(allowed.tolist()))
            m["campaign_split"]=split;m["campaign_scope"]="Custom SPT-JEPA manifest using pinned CLEAR v0.8 task semantics; reference weights may have trained on these episodes."
            m["metadata_card_sha256"]=hashlib.sha256((root/"dataset_card.json").read_bytes()).hexdigest()
            manifests.save_manifest(m,out/(split+"-"+mode+".json"));print(split,mode,m["statistics"],flush=True)
if __name__=="__main__":main()
