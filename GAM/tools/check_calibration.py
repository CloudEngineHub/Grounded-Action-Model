"""Check the camera calibration and the robot URDF on a recorded episode: the URDF silhouette, projected with
your intrinsics / extrinsics from the recorded joint state, is drawn over the camera image.

    python -m GAM.tools.check_calibration --config my_task.yaml [--episode 0] [--frames 0,100,200] [--out calib_check]

If the outline hugs the robot in every frame, K, E (base -> camera) and the URDF joint mapping are right; a shifted or
scaled outline means a wrong extrinsic / intrinsic, a wrong joint order or a wrong T_base_root. Nothing else of the
pipeline is needed (no detector, no GPU).
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from GAM.config import load_config
from GAM.data.episodes import list_episodes, load_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", default="0,mid,last", help="frame indices, 'mid' / 'last' allowed")
    ap.add_argument("--out", default="calib_check")
    a = ap.parse_args()
    cfg = load_config(a.config)
    assert cfg["perception"].get("robot"), "perception.robot (urdf, joint_names, T_base_root) is not set in the config"
    from GAM.precompute import build_arm_mask_fn
    arm_fn = build_arm_mask_fn(cfg)
    cfg_noclick = dict(cfg, data=dict(cfg["data"], clicks={"source": "none"}))
    path = list_episodes(cfg_noclick)[a.episode]; ep = load_episode(path, cfg_noclick)
    os.makedirs(a.out, exist_ok=True)
    for tok in a.frames.split(","):
        t = {"mid": ep.T // 2, "last": ep.T - 1}.get(tok.strip(), None)
        t = int(tok) if t is None else t
        rgb = ep.rgb(t); H, W = rgb.shape[:2]
        mask = np.asarray(arm_fn(ep.state[t].astype(np.float64), ep.K[t].astype(np.float64), ep.E[t].astype(np.float64), H, W), bool)
        img = np.ascontiguousarray(rgb[:, :, ::-1]).copy()                       # BGR for cv2
        img[mask] = (0.55 * img[mask] + 0.45 * np.array([0, 200, 255])).astype(np.uint8)   # tint the silhouette
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 255), 2)
        out = os.path.join(a.out, f"episode{a.episode}_f{t}.png"); cv2.imwrite(out, img)
        print(f"frame {t}: silhouette covers {100 * mask.mean():.1f}% of the image -> {out}", flush=True)
    ep.close()
    print("[check_calibration] done: the outline should hug the robot in every frame")


if __name__ == "__main__":
    main()
