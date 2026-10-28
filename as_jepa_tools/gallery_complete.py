"""Complete fixed/best/worst gallery metadata from the frozen quality RGB banks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from as_jepa_tools.data import PushTWindows
from as_jepa_tools.quality_complete import video


def main():
    parser=argparse.ArgumentParser()
    for name in ["quality","checkpoint","data","output"]:
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--planning")
    args=parser.parse_args()
    quality,checkpoint,out=Path(args.quality),Path(args.checkpoint),Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():
        raise RuntimeError("Completed gallery already exists; preserve it")
    cases=json.loads((quality/"cases.json").read_text())
    cfg=json.loads((checkpoint/"config.json").read_text())
    weights_hash=hashlib.sha256((checkpoint/"weights.pt").read_bytes()).hexdigest()
    quality_protocol=json.loads((quality/"metrics.json").read_text())
    pixel_feedback=any(name in cfg.get("_target_","") for name in ["ActionRepWorld","SemanticCarry"])
    metrics=np.load(quality/"per-case-metrics.npz")
    truth=np.load(quality/"truth-rgb.npy",mmap_mode="r")
    mean=np.load(quality/"mean-rgb.npy",mmap_mode="r")
    samples=np.load(quality/"samples-rgb.npy",mmap_mode="r") if (quality/"samples-rgb.npy").exists() else None
    data=PushTWindows(args.data,window=16,gap=8,split="test")
    data.open()
    score=.5*(metrics["mean_agent"].mean(1)+metrics["mean_block"].mean(1))
    groups=dict(fixed=list(range(min(32,len(cases)))),best=np.argsort(score)[:8].tolist(),
                worst=np.argsort(score)[-8:][::-1].tolist())
    for label,selected in groups.items():
        directory=out/label
        directory.mkdir(exist_ok=True)
        review=[]
        for position,index in enumerate(selected):
            record=cases[index]
            folder=directory/str(position).zfill(2)
            folder.mkdir(exist_ok=True)
            gt=np.array(truth[index,3:])
            prediction=np.array(mean[index])
            copy=np.repeat(np.array(truth[index,2:3]),13,axis=0)
            rows=[("truth",gt),("mean",prediction)]
            if samples is not None:
                rows += [(f"sample{s}",np.array(samples[index,s])) for s in range(samples.shape[1])]
            rows.append(("copy",copy))
            video(folder/"gt.mp4",gt)
            video(folder/"pred_mean.mp4",prediction)
            video(folder/"copy.mp4",copy)
            if samples is not None:
                for sample in range(samples.shape[1]):
                    video(folder/f"pred_sample_{sample}.mp4",np.array(samples[index,sample]))
            grid=Image.new("RGB",(80+13*64,len(rows)*64+22),"white")
            draw=ImageDraw.Draw(grid)
            for horizon in range(13):draw.text((80+horizon*64+3,3),str(horizon+1),fill="black")
            for row,(name,frames) in enumerate(rows):
                draw.text((3,25+row*64),name,fill="black")
                for horizon,frame in enumerate(frames):
                    grid.paste(Image.fromarray(frame),(80+horizon*64,22+row*64))
            grid.save(folder/"grid.png")
            episode,start=int(record["episode"]),int(record["start"])
            root_row=int(data.offsets[episode])+start
            anchor_gap=cfg.get("anchor_gap",0)
            anchor_row=root_row-anchor_gap
            per_case={key:np.asarray(metrics[key][index]).tolist() for key in metrics.files}
            batch_size=quality_protocol.get("evaluation_batch_size",8)
            batch_index=index//batch_size;within=index%batch_size
            mean_seeds=([1000+batch_index]*13 if pixel_feedback else [1000+batch_index*100+((within*13+h)//64)*64 for h in range(13)])
            clean_record={key:value for key,value in record.items() if key!="mean_renderer_seed"}
            metadata=dict(**clean_record,bank_case_index=index,model_config=cfg,checkpoint=str(checkpoint),
                          checkpoint_sha256=weights_hash,anchor_row=anchor_row,
                          context_rows=list(range(root_row,root_row+3)),anchor_raw_row=int(data.raw_rows[anchor_row]),
                          target_rows=list(range(root_row+3,root_row+16)),
                          normalized_actions=np.array(data.actions[root_row:root_row+16]).tolist(),
                          per_horizon_errors=per_case,mean_pose_error_px=float(score[index]),
                          selection=label,selection_scope="First32fixedcases;best/worst8bymeanagentandblockposeerroroverall13horizons.",
                          row_labels=[name for name,_ in rows],samples=0 if samples is None else samples.shape[1],
                          model_noise_seeds=dict(mean_renderer_by_horizon=mean_seeds,evaluation_batch_size=batch_size,case_within_batch=within,source_nominal_renderer_seed=record.get("mean_renderer_seed"),
                                                 sampled_latent_base=20261027+int(record["index"])*101,
                                                 sampled_renderer_base=31000+int(record["index"])*13),
                          source_quality=str(quality),forward_protocol=quality_protocol["forward_precision"],seed_note="Nativepixelrolloutuses1000+batch_index;latentmeanrenderinguses1000+100*batch_index+chunk_offset.LegacyqualitycasesrecordedthenominallatentbasealsoforNative;actualpolicypreservedhere.")
            (folder/"meta.json").write_text(json.dumps(metadata,indent=2))
            review.append(f"{position:02d},{episode},{start},pending,")
        (directory/"review.csv").write_text("case,episode,start,pass_fail,reason\n"+"\n".join(review)+"\n")
    planning_records=[]
    if args.planning:
        planning_root=Path(args.planning)
        counts={"success":0,"failure":0}
        for metric_path in sorted(planning_root.glob("seed*/metrics.json")):
            record=json.loads(metric_path.read_text())
            for env_index,success in enumerate(record["metrics"]["episode_successes"]):
                label="success" if success else "failure"
                if counts[label]>=5:continue
                original=metric_path.parent/f"env_{env_index}.mp4"
                if not original.exists():raise FileNotFoundError(original)
                destination=out/"planning"/label/f"{counts[label]:02d}.mp4"
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():destination.hardlink_to(original)
                planning_records.append(dict(outcome=label,episode=int(record["episodes"][env_index]),
                                             start=int(record["start_steps"][env_index]),evaluation_seed=record["seed"],
                                             source_video=str(original),gallery_video=str(destination),environment_index=env_index,
                                             checkpoint=str(checkpoint),checkpoint_sha256=weights_hash))
                counts[label]+=1
        (out/"planning").mkdir(exist_ok=True)
        (out/"planning/index.json").write_text(json.dumps(dict(videos=planning_records,counts=counts,scope="Firstfiveavailable successes/failures,fromexistingexactP0episodes;noadditionalplanningruns."),indent=2))
    (out/"complete.json").write_text(json.dumps(dict(groups=groups,planning_video_count=len(planning_records),checkpoint=str(checkpoint),
                                                    checkpoint_sha256=weights_hash,human_review_complete=False,
                                                    scope="Completefixed/best/worstmediaandmetadatafromsaveduint8banks;userreviewremainsrequired.Counterfactual/swap/planningmediahaveindependentdirectories."),indent=2))
    print(json.dumps(dict(fixed=len(groups["fixed"]),best=8,worst=8,human_review_complete=False)))


if __name__=="__main__":
    main()
