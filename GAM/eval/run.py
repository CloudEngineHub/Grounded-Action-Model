"""Evaluate a GAM checkpoint closed-loop in any environment, yours, or the built-in replay of recorded episodes.

    python -m GAM.eval.run --config my_task.yaml --ckpt runs/my_task/latest.pt --env my_robot.py:MyEnv --episodes 20
    python -m GAM.eval.run --config my_task.yaml --ckpt runs/my_task/latest.pt --env replay --episodes 5   # recorded episodes, open loop

The environment is a small adapter class (see GAM/eval/envs.py for the protocol and the replay example):
    env = MyEnv(cfg)                      # the yaml, for whatever the adapter needs (data paths, robot)
    instruction = env.reset(episode)      # new episode -> its task string ("" if none)
    obs = env.observe()                   # {"rgb": (H,W,3) uint8 RGB, "depth": (H,W) m or mm, "K": (3,3), "E": (3,4) base->cam, "state": (Dq,)}
    done, success = env.step(action)      # execute one action (the checkpoint's action layout)
    env.close()
Every episode runs GAMRunner (perception + policy) through ActionChunker (eval.chunking: replan | tempens | rtc),
for at most eval.max_steps steps. Written to --out: results.jsonl (one line per episode), and with --video the camera
frames with the detector's 3D boxes drawn (episode<i>_<S|F>.mp4).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

from GAM.config import load_config
from GAM.eval.chunker import ActionChunker
from GAM.eval.runner import GAMRunner
from GAM.perception.draw import draw_boxes


def load_env_class(spec: str):
    """'replay' | 'module.path:Class' | 'path/to/file.py:Class'"""
    if spec == "replay":
        from GAM.eval.envs import ReplayEnv
        return ReplayEnv
    mod, _, cls = spec.partition(":")
    assert cls, "--env takes module:Class or file.py:Class"
    if mod.endswith(".py"):
        s = importlib.util.spec_from_file_location("gam_user_env", mod); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    else:
        sys.path.insert(0, os.getcwd()); m = importlib.import_module(mod)
    return getattr(m, cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", default=None, help="default: checkpoints.policy")
    ap.add_argument("--env", default="replay", help="replay | module:Class | file.py:Class (see GAM/eval/envs.py)")
    ap.add_argument("--episodes", type=int, default=None, help="default: eval.episodes (replay: all recorded episodes)")
    ap.add_argument("--max-steps", type=int, default=None, help="default: eval.max_steps (replay: the episode length)")
    ap.add_argument("--chunking", choices=["replan", "tempens", "rtc"], default=None, help="default: eval.chunking")
    ap.add_argument("--gripper-dims", default=None, help="action dims never blended / averaged across chunks (default: eval.gripper_dims, else 6,13)")
    ap.add_argument("--out", default=None, help="default: eval.out_dir")
    ap.add_argument("--video", action="store_true", help="write episode<i>_<S|F>.mp4 with the 3D boxes the policy saw")
    a = ap.parse_args()
    cfg = load_config(a.config); ev = dict(cfg.get("eval", {}))
    if a.chunking:
        ev["chunking"] = a.chunking
    out = a.out or ev.get("out_dir") or "./runs/eval"; os.makedirs(out, exist_ok=True)
    grip = tuple(int(x) for x in a.gripper_dims.split(",") if x.strip()) if a.gripper_dims else tuple(ev.get("gripper_dims", (6, 13)))
    runner = GAMRunner(cfg, a.ckpt or cfg["checkpoints"]["policy"]); chunker = ActionChunker.from_config(ev, gripper_dims=grip)
    Env = load_env_class(a.env); env = Env(cfg)
    n = a.episodes or int(ev.get("episodes", getattr(env, "n_episodes", 10) or 10)); n = min(n, getattr(env, "n_episodes", n) or n)
    max_steps = a.max_steps or int(ev.get("max_steps", 0)) or None
    res_p = os.path.join(out, "results.jsonl"); succ = 0
    print(f"[run] {a.env}: {n} episodes, chunking={chunker.mode}, max_steps={max_steps or 'episode length'} -> {res_p}", flush=True)
    for ep in range(n):
        instruction = env.reset(ep) or ""; runner.reset(instruction, seed=ep); chunker.reset()
        frames, errs, steps, done, ok, t0 = [], [], 0, False, False, time.time()
        limit = max_steps or getattr(env, "T", None) or 10 ** 9
        while not done and steps < limit:
            obs = env.observe()
            if not chunker.needs_query():
                runner.advance(obs["rgb"])                               # text+track: the tracker sees every frame
            a_t = chunker.act(lambda prefix: runner.act(obs["rgb"], obs["depth"], obs["K"], obs["E"], obs["state"], prefix)[0])
            if a.video:
                boxes = runner.boxes(obs["rgb"], obs["depth"], obs["K"], obs["E"], obs["state"])
                frames.append(draw_boxes(np.ascontiguousarray(obs["rgb"]).copy(), boxes, obs["K"], obs["E"], runner.slots))
            if "action" in obs:                                          # replay: recorded action for the same step
                errs.append(np.abs(np.asarray(a_t, np.float32) - np.asarray(obs["action"], np.float32)))
            done, ok = env.step(a_t); steps += 1
        rec = {"episode": ep, "instruction": instruction, "steps": steps, "success": bool(ok), "secs": round(time.time() - t0, 1)}
        if errs:
            rec["action_mae"] = float(np.mean(errs)); rec["action_mae_per_dim"] = np.mean(errs, 0).round(4).tolist()
        succ += bool(ok); open(res_p, "a").write(json.dumps(rec) + "\n")
        if frames:
            import imageio
            imageio.mimwrite(os.path.join(out, f"episode{ep}_{'S' if ok else 'F'}.mp4"), frames, fps=10, quality=7)
        print(f"  ep{ep}: {'success' if ok else 'fail'} steps={steps}" + (f" action_mae={rec['action_mae']:.4f}" if errs else "") + f" ({rec['secs']}s) running {succ}/{ep + 1}", flush=True)
    env.close()
    print(f"[run] done: {succ}/{n} = {100.0 * succ / max(n, 1):.1f}%", flush=True)


if __name__ == "__main__":
    main()
