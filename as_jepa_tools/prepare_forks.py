"""Training-only simulator interventions with independent held-out episodes."""
import argparse,json,io,time
from pathlib import Path
import numpy as np
from PIL import Image
from as_jepa_tools.data import PushTWindows
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",required=True);p.add_argument("--cases",type=int,default=1024);a=p.parse_args()
 from stable_worldmodel.envs.pusht import PushT
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 data=PushTWindows(a.data,window=16,gap=8,split="train");data.open();rng=np.random.default_rng(20261028)
 card=json.loads((Path(a.data)/"dataset_card.json").read_text());mean=np.array(card["action_mean"]);std=np.array(card["action_std"])
 candidates=rng.permutation(len(data));eligible=[int(x) for x in candidates if data.locate(x)[1]>=8][:a.cases]
 contexts=np.lib.format.open_memmap(out/"contexts.npy",mode="w+",dtype=np.uint8,shape=(len(eligible),224,224,3))
 futures=np.lib.format.open_memmap(out/"futures.npy",mode="w+",dtype=np.uint8,shape=(len(eligible),4,224,224,3))
 actions=np.lib.format.open_memmap(out/"actions.npy",mode="w+",dtype=np.float32,shape=(len(eligible),4,8,10))
 records=[];env=PushT(render_mode="rgb_array")
 for i,index in enumerate(eligible):
  batch=data[index];ep,start=data.locate(index);origin=int(batch["rows"][2]);state=np.array(data.states[origin])
  contexts[i]=np.array(data.p224[origin])
  for branch in range(4):
   donor=batch if branch==0 else data[eligible[int(rng.integers(len(eligible)))]]
   sequence=donor["action"][2:10].numpy().reshape(40,2)*std+mean
   env.reset(seed=i);env._set_state(state.copy())
   for action in sequence:env.step(action)
   futures[i,branch]=np.asarray(Image.fromarray(env.render()).resize((224,224),Image.Resampling.BOX))
   actions[i,branch]=((sequence-mean)/std).reshape(8,10)
  records.append(dict(episode=int(ep),start=int(start),origin_row=origin))
  if i%32==0:(out/"status.json").write_text(json.dumps(dict(state="running",completed=i+1,total=len(eligible),updated_at=time.time())))
 env.close();contexts.flush();futures.flush();actions.flush()
 episodes=np.array([r["episode"] for r in records]);validation_ep=np.unique(episodes)[::5];train=np.flatnonzero(~np.isin(episodes,validation_ep));validation=np.flatnonzero(np.isin(episodes,validation_ep))
 np.save(out/"train_indices.npy",train);np.save(out/"validation_indices.npy",validation)
 (out/"cases.json").write_text(json.dumps(records,indent=2));(out/"metrics.json").write_text(json.dumps(dict(cases=len(eligible),train=len(train),validation=len(validation),episode_overlap=0,scope="Forks only from original train episodes; simulator-state restoration lacks saved block velocity,allbranches use identical resets; no branchID/seed/palette input to classifier."),indent=2))
 (out/"status.json").write_text(json.dumps(dict(state="finished",completed=len(eligible),updated_at=time.time())))
if __name__=="__main__":main()
