"""Convert RoboTwin demonstrations (the HDF5 files of collect_data.sh) to a LeRobot dataset.

    python benchmarks/robotwin/to_lerobot.py --data $ROBOTWIN_DATA --task shake_bottle --out ./lerobot/robotwin_shake_bottle \\
        [--instruction "shake the bottle"] [--fps 15] [--push user/repo]

Per frame: observation.images.head (RGB), observation.depth (uint16 mm, optional with --depth), observation.state
and action (14-d joint vectors), the task string. The still frames listed in frames_keep.json are dropped, as in
GAM.precompute. Needs `pip install lerobot`.
"""
import argparse, glob, json, os

import cv2
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--task", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--instruction", default=None, help="task string (default: the task name with spaces)")
    ap.add_argument("--fps", type=int, default=15); ap.add_argument("--depth", action="store_true", help="also store the depth (uint16 mm)")
    ap.add_argument("--max-episodes", type=int, default=0); ap.add_argument("--push", default=None, help="Hub repo id to push to")
    a = ap.parse_args()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    files = sorted(glob.glob(os.path.join(a.data, a.task, "gam_collect", "data", "episode*.hdf5")),
                   key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()) or 0))
    if a.max_episodes:
        files = files[:a.max_episodes]
    keep = json.load(open(os.path.join(a.data, "frames_keep.json"))).get(a.task, {}) if os.path.exists(os.path.join(a.data, "frames_keep.json")) else {}
    instruction = a.instruction or a.task.replace("_", " ")
    with h5py.File(files[0], "r") as h:
        rgb0 = h["observation/head_camera/rgb"][0]; rgb0 = cv2.imdecode(np.frombuffer(np.asarray(rgb0).tobytes(), np.uint8), cv2.IMREAD_COLOR) if np.asarray(rgb0).ndim < 3 else np.asarray(rgb0)
        H, W = rgb0.shape[:2]; D = h["joint_action/vector"].shape[1]
        K = np.asarray(h["observation/head_camera/intrinsic_cv"][0], np.float64).reshape(3, 3).tolist()
        E = np.asarray(h["observation/head_camera/extrinsic_cv"][0], np.float64).reshape(-1)[:12].reshape(3, 4).tolist()
    features = {"observation.images.head": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
                "observation.state": {"dtype": "float32", "shape": (D,), "names": None},
                "action": {"dtype": "float32", "shape": (D,), "names": None}}
    if a.depth:
        features["observation.depth"] = {"dtype": "uint16", "shape": (H, W), "names": None}
    assert not os.path.exists(a.out), f"{a.out} exists; remove it first"
    ds = LeRobotDataset.create(os.path.basename(os.path.abspath(a.out)), fps=a.fps, root=a.out, features=features, use_videos=True)
    for f in files:
        ep = "".join(ch for ch in os.path.splitext(os.path.basename(f))[0] if ch.isdigit())
        with h5py.File(f, "r") as h:
            q = np.asarray(h["joint_action/vector"], np.float32); T = len(q)
            idx = np.asarray(keep.get(ep, {}).get("frames_keep", range(T)), int)
            for t in idx:
                rgb = h["observation/head_camera/rgb"][t]
                rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(np.asarray(rgb).tobytes(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if np.asarray(rgb).ndim < 3 else np.asarray(rgb)
                frame = {"observation.images.head": rgb, "observation.state": q[t], "action": q[t], "task": instruction}
                if a.depth:
                    frame["observation.depth"] = np.asarray(h["observation/head_camera/depth"][t]).astype(np.uint16)
                ds.add_frame(frame)
        ds.save_episode()
        print(f"episode {ep}: {len(idx)}/{T} frames", flush=True)
    if hasattr(ds, "finalize"):                                  # lerobot >= 0.4 writes the metadata / video index here
        ds.finalize()
    print(f"dataset written to {a.out}: {ds.num_episodes} episodes, {ds.num_frames} frames")
    print("camera_intrinsics:", json.dumps(K)); print("camera_extrinsics:", json.dumps(E))
    if a.push:
        ds.push_to_hub(a.push, private=True); print(f"pushed to {a.push}")


if __name__ == "__main__":
    main()
