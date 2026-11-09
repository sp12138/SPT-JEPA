"""Replay a source transition to verify RGB orientation and camera geometry.

Simulator state/XML are diagnostic-only and never returned to a learner.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.robocasa_compare.acquire import save
from experiments.robocasa_compare.evaluate import install_public


def verify(root):
    import av
    import cv2
    import pyarrow.parquet as pq
    from robots.robocasa.robot_env import RoboCasaEnv
    cv2.setNumThreads(1)
    source = root / "data/OpenDrawer/lerobot"
    extra = source / "extras/episode_000000"
    metadata = json.loads((extra / "ep_meta.json").read_text())
    wrapper = RoboCasaEnv(task_name="OpenDrawer", camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
        camera_width=320, camera_height=240, seed=0, robot="PandaOmron", chunk_size=4,
        wm_embodiment="franka_omron_state_delta", kitchen_split="pretrain",
        layout_and_style_ids=[(metadata["layout_id"], metadata["style_id"])])
    try:
        actual = wrapper._env
        actual.set_ep_meta(metadata)
        actual.reset()
        xml = actual.edit_model_xml(gzip.open(extra / "model.xml.gz", "rt").read())
        actual.reset_from_xml_string(xml)
        actual.sim.reset()
        actual.sim.set_state_from_flattened(np.load(extra / "states.npz")["states"][0])
        actual.sim.forward()
        observation = wrapper._build_obs()
        sensors = actual._get_observations(force_update=True)
        sensor_values = {k: np.asarray(v).tolist() for k,v in sensors.items()
                         if "image" not in k and any(x in k for x in ("base_pos", "base_quat", "eef_pos", "eef_quat"))}
        recorded = np.asarray(pq.read_table(source / "data/chunk-000/episode_000000.parquet",
                              columns=["observation.state"])["observation.state"].to_pylist())
        recorded = np.concatenate([recorded[:, :13], recorded[:, 14:15]], axis=1)
        original = recorded[0]
        # Quaternions may differ by a global sign; translation is unambiguous.
        translation_error = float(np.max(np.abs(observation["proprio_robocasa"][[0,1,2,7,8,9]] - original[[0,1,2,7,8,9]])))
        aligned = np.concatenate([sensors["robot0_base_pos"], sensors["robot0_base_quat"],
                                  sensors["robot0_base_to_eef_pos"], sensors["robot0_base_to_eef_quat"][:3],
                                  sensors["robot0_gripper_qpos"][:1]])
        aligned_error = float(np.max(np.abs(aligned-original)))
        rows = []
        for index, camera in enumerate(("robot0_agentview_left", "robot0_eye_in_hand")):
            with av.open(str(source / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4")) as stream:
                video = next(stream.decode(video=0)).to_ndarray(format="rgb24")
            square = actual.sim.render(height=256, width=256, camera_name=camera)[::-1].copy()
            rectangle = observation[f"img_{index}"]
            cropped = rectangle[:, (320-240)//2:(320+240)//2]
            resize = lambda image: cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA).astype(float)
            mse = lambda a,b: float(np.mean((resize(a)-resize(b))**2))
            rows.append(dict(camera=camera, video_shape=list(video.shape),
                             stored_video_vs_replayed_square_mse=mse(video,square),
                             stored_video_vs_flipped_replay_mse=mse(video,square[::-1]),
                             square_vs_resized_rectangle_mse=mse(square,rectangle),
                             square_vs_center_cropped_rectangle_mse=mse(square,cropped)))
        actual.sim.set_state_from_flattened(np.load(extra / "states.npz")["states"][4])
        actual.sim.forward()
        next_sensors = actual._get_observations(force_update=True)
        next_aligned = np.concatenate([next_sensors["robot0_base_pos"], next_sensors["robot0_base_quat"],
                                       next_sensors["robot0_base_to_eef_pos"], next_sensors["robot0_base_to_eef_quat"][:3],
                                       next_sensors["robot0_gripper_qpos"][:1]])
        delta_error = float(np.max(np.abs((next_aligned-aligned) - (recorded[4]-recorded[0]))))
        result = dict(passed=aligned_error < 1e-5 and delta_error < 1e-5 and all(
            x["stored_video_vs_replayed_square_mse"] < x["stored_video_vs_flipped_replay_mse"] for x in rows),
            translation_alignment_max_m=translation_error, replayed_pose=observation["proprio_robocasa"].tolist(),
            stored_pose=original.tolist(), rows=rows,
            native_sensor_pose=aligned.tolist(), native_sensor_alignment_max=aligned_error,
            factual_four_native_step_input_alignment_max=delta_error,
            pose_sensors=sensor_values,
            diagnostic_only=True, privileged_state_does_not_enter_learner=True)
        save(root / "data-view-replay.json", result)
        print(json.dumps(result), flush=True)
        assert result["passed"], result
    finally:
        wrapper.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    install_public(args.root)
    verify(args.root)
