"""Offline object tracks for end-to-end training on LIBERO (the only thing computed ahead of time).

    python benchmarks/libero_pro/track.py --config configs/libero_pro.yaml --suite object [--tasks 0,3] [--max-demos 50]

Per demonstration: replay the kept frames in the simulator (render only), detect the instruction's
objects on frame 0 (text prompts, WildDet3D) and follow them through the whole demonstration with
the offline CoTracker model. Output: <train.tracks_dir>/libero_<suite>__<task>/<demo>.npz with
  points (T,4,2) px per slot (NaN = not visible), kept (T,) original frame indices, instruction.
A few KB per demonstration. `python -m GAM.train --config configs/libero_pro.yaml --e2e` reads them.
"""
import argparse, glob, os, re, sys

os.environ.setdefault("MUJOCO_GL", "egl")
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from GAM.config import load_yaml
from GAM.hub import fetch
from eval import task_dirs
from frame_builder import FrameBuilder
from precompute import noop_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", choices=["spatial", "object", "goal", "10"], required=True)
    ap.add_argument("--tasks", default=None); ap.add_argument("--max-demos", type=int, default=0)
    ap.add_argument("--no-clean", action="store_true", help="keep no-op frames")
    args = ap.parse_args()
    cfg = load_yaml(args.config); pc, ck, dt = cfg["perception"], cfg["checkpoints"], cfg["data"]
    res = int(pc.get("image_size", 512))
    bddl_dir, _ = task_dirs(cfg, f"libero_{args.suite}")
    demo_dir = os.path.join(dt["demos_root"], f"libero_{args.suite}")
    assert os.path.isdir(demo_dir), f"{demo_dir} not found: set data.demos_root (LIBERO_DATASETS)"
    wd_ckpt, tracker_ckpt, offline_ckpt = fetch(ck["wilddet3d"]), fetch(ck["cotracker"]), fetch(ck["cotracker_offline"])
    from libero.libero.envs import OffScreenRenderEnv
    bddls = sorted(glob.glob(os.path.join(bddl_dir, "*.bddl")))
    sel = [int(x) for x in args.tasks.split(",")] if args.tasks else range(len(bddls))
    fb = None
    for ti in sel:
        bddl = bddls[ti]; name = os.path.splitext(os.path.basename(bddl))[0]
        instruction = " ".join(re.search(r"\(:language\s+([^)]*)\)", open(bddl).read()).group(1).split())
        out_dir = os.path.join(cfg["train"]["tracks_dir"], f"libero_{args.suite}__{name}"); os.makedirs(out_dir, exist_ok=True)
        print(f"[task {ti}] {name}: {instruction!r} -> {out_dir}", flush=True)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res, camera_depths=True)
        env.seed(0); env.reset(); sim = env.env.sim
        if fb is None:
            fb = FrameBuilder(sim, dict(pc, span_head=ck.get("span_head")), wd_ckpt, tracker_ckpt)
        fb.set_task(instruction)
        with h5py.File(os.path.join(demo_dir, f"{name}_demo.hdf5"), "r") as f:
            demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
            if args.max_demos:
                demos = demos[:args.max_demos]
            for di, dk in enumerate(demos):
                outp = os.path.join(out_dir, f"{dk}.npz")
                if os.path.exists(outp):
                    continue
                g = f[f"data/{dk}"]; states = g["states"][:]; actions = g["actions"][:].astype(np.float32)
                idx = np.nonzero(np.ones(len(states), bool) if args.no_clean else noop_mask(actions))[0]
                fb.reset(sim)
                frames = [np.flipud(env.set_init_state(states[t])["agentview_image"]).copy() for t in idx]
                pts0, _ = fb.ground_first_frame(sim, env.set_init_state(states[idx[0]]))
                slots = [k for k in range(4) if np.isfinite(pts0[k]).all()]
                points = np.full((len(idx), 4, 2), np.nan, np.float32)
                if slots:
                    tr, vis = fb.track_offline(frames, [tuple(pts0[k]) for k in slots], offline_ckpt)
                    for i, k in enumerate(slots):
                        ok = vis[:, i] & (tr[:, i, 0] >= 0) & (tr[:, i, 0] < res) & (tr[:, i, 1] >= 0) & (tr[:, i, 1] < res)
                        points[ok, k] = tr[ok, i]
                with open(outp + ".tmp", "wb") as fh: np.savez_compressed(fh, points=points, kept=idx.astype(np.int32), instruction=instruction,
                                    prompts=np.array([p or "" for p in fb.prompts]))
                os.replace(outp + ".tmp", outp)
                vis_rate = {fb.prompts[k]: round(float(np.isfinite(points[:, k, 0]).mean()), 2) for k in slots}
                print(f"  [{di + 1}/{len(demos)}] {dk}: {len(idx)} frames, visible {vis_rate}", flush=True)
        env.close()
    print("[track] done", flush=True)


if __name__ == "__main__":
    main()
