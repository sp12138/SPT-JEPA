"""Render paired GT/prediction videos and temporal sheets for a finite SPT demo."""
import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("run_root", type=Path)
parser.add_argument("sample_dir", type=Path)
parser.add_argument("--previous", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.run_root / "source/runtime"))
from vforge.tasks.rep_world.data import read_video_clip

cfg = OmegaConf.load(args.run_root / "source/jobs/configs/pretrain_50k.yaml")
rows = [json.loads(line) for line in Path(cfg.data.val_manifest).read_text().splitlines()]
records = {row["sample_id"]: row for row in rows}
videos = sorted(args.sample_dir.glob("*.mp4"))
assert len(videos) == 8
old = args.previous
pairs = []
summary = []
steps = set()
previous_steps = set()


def decode(path):
    with av.open(str(path)) as handle:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in handle.decode(video=0)])


for path in videos:
    sidecar = json.loads(Path(str(path) + ".json").read_text())
    sample_id = sidecar["sample_id"]
    record = records[sample_id]
    prediction = decode(path)
    truth_tensor = read_video_clip(record["rgb_path"], 144, 1, random_clip=False,
                                   start_frame=record["start_frame"])
    truth = truth_tensor.clamp(-1, 1).add(1).mul(127.5).round().to(dtype=torch.uint8)
    truth = truth.permute(1, 2, 3, 0).numpy()
    assert prediction.shape == truth.shape == (144, 256, 256, 3)
    previous = decode(old / path.name)
    old_sidecar = json.loads((old / (path.name + ".json")).read_text())
    assert sidecar["seed"] == old_sidecar["seed"], sample_id
    assert sidecar["sample_id"] == old_sidecar["sample_id"], sample_id
    assert sidecar["metadata"]["evaluation_weights"] == "ema"
    step = int(sidecar["metadata"]["ema_updates"])
    previous_step = int(old_sidecar["metadata"]["ema_updates"])
    assert step > 0 and old_sidecar["metadata"]["evaluation_weights"] == "ema"
    assert sidecar["metadata"]["steps"] == 50 and sidecar["metadata"]["cfg_scale"] == 1
    steps.add(step)
    previous_steps.add(previous_step)
    pred_delta = np.mean(np.abs(np.diff(prediction[16:].astype(np.float32), axis=0)))
    gt_delta = np.mean(np.abs(np.diff(truth[16:].astype(np.float32), axis=0)))
    summary.append(dict(sample_id=sample_id, file=path.name, frames=144,
                        seed=sidecar["seed"], weights="ema", step=step,
                        generated_frame_delta_8bit=float(pred_delta),
                        reference_frame_delta_8bit=float(gt_delta),
                        previous_step=previous_step, same_case_seed_as_previous=True))
    pairs.append((truth, prediction, previous))

assert len(steps) == len(previous_steps) == 1
step = steps.pop()
previous_step = previous_steps.pop()
render = args.sample_dir / "review"
render.mkdir(exist_ok=True)
for group in range(2):
    size = 160
    row_height = size + 28
    sheet = Image.new("RGB", (7 * size, 4 * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    for offset, (truth, prediction, previous) in enumerate(pairs[group * 4:group * 4 + 4]):
        case = group * 4 + offset + 1
        entries = [(truth[0], "Input f0"), (prediction[16], "Pred f16"),
                   (prediction[47], "Pred f47"), (prediction[95], "Pred f95"),
                   (prediction[143], "Pred f143"), (truth[143], "GT f143"),
                   (previous[143], f"Step{previous_step} f143")]
        for col, (pixels, caption) in enumerate(entries):
            sheet.paste(Image.fromarray(pixels).resize((size, size)),
                        (col * size, offset * row_height))
            draw.text((col * size + 4, offset * row_height + size + 5),
                      f"{case}: {caption}", fill="black")
    sheet.save(render / f"temporal_cases_{group * 4 + 1}_{group * 4 + 4}.jpg", quality=90)

container = av.open(str(render / "gt_vs_prediction_8cases.mp4"), "w")
stream = container.add_stream("libx264", rate=16)
stream.width, stream.height, stream.pix_fmt = 640, 752, "yuv420p"
stream.options = {"crf": "20", "preset": "fast"}
for n in range(144):
    canvas = Image.new("RGB", (640, 752), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (truth, prediction, _) in enumerate(pairs):
        x, y = (index % 2) * 320, (index // 2) * 188
        canvas.paste(Image.fromarray(truth[n]).resize((160, 160)), (x, y))
        canvas.paste(Image.fromarray(prediction[n]).resize((160, 160)), (x + 160, y))
        draw.text((x + 4, y + 162), f"{index+1} GT | EMA {step}  f{n}", fill="black")
    frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
for packet in stream.encode():
    container.mux(packet)
container.close()
(render / "review.json").write_text(json.dumps(dict(
    step=step, previous_step=previous_step, samples=summary, protocol=dict(history_frames=16, generated_frames=128,
    seed=20260920, steps_per_state=50, cfg=1, fps=16, weights="ema")), indent=2) + "\n")
print(json.dumps(dict(samples=len(pairs), review=str(render)), indent=2))
