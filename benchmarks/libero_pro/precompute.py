"""Build the GAM training cache for LIBERO from the demonstration files.

    python benchmarks/libero_pro/precompute.py --config configs/libero_pro.yaml --suite object [--tasks 0,3] [--max-demos 50]

Each demonstration is replayed from its recorded simulator states; every kept frame goes through the
same FrameBuilder as the evaluation (text-prompted detection, robot mask from the robot model, object
geometry from the detector), so train and test see identical features. The object points are
followed through the whole demonstration with the *offline* CoTracker model (it sees the full video,
which is more accurate than the online tracker used at test time).

No simulator ground truth is used: inputs are the agentview RGB-D, the camera calibration, the joint
state and the instruction. Output: <train.cache_dir>/<suite>__<task>/<demo>.npz, loadable by GAM.train.
"""
import argparse, glob, os, re, sys, time

os.environ.setdefault("MUJOCO_GL", "egl")
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from GAM.config import load_yaml
from GAM.hub import fetch
from frame_builder import FrameBuilder, SLOTS


def noop_mask(actions):
    """OpenVLA's rule: drop frames with zero arm motion and an unchanged gripper command."""
    keep = np.ones(len(actions), bool)
    for t in range(len(actions)):
        still = np.abs(actions[t, :6]).max() < 1e-6
        same_grip = t > 0 and abs(actions[t, 6] - actions[t - 1, 6]) < 1e-6
        keep[t] = not (still and same_grip)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", choices=["spatial", "object", "goal", "10"], required=True)
    ap.add_argument("--tasks", default=None, help="comma-separated task indices (default: all 10)")
    ap.add_argument("--max-demos", type=int, default=0, help="per task (default: all)")
    ap.add_argument("--no-clean", action="store_true", help="keep no-op frames")
    args = ap.parse_args()
    cfg = load_yaml(args.config); pc, ck, dt = cfg["perception"], cfg["checkpoints"], cfg["data"]
    res = int(pc.get("image_size", 512)); G = int(pc.get("patch_grid", 6)); n_pts = int(pc.get("n_points", 512))
    from eval import task_dirs
    bddl_dir, _ = task_dirs(cfg, f"libero_{args.suite}")
    demo_dir = os.path.join(dt["demos_root"], f"libero_{args.suite}")
    assert os.path.isdir(demo_dir), f"{demo_dir} not found: set data.demos_root (LIBERO_DATASETS) to the LIBERO demonstrations"
    wd_ckpt, tracker_ckpt = fetch(ck["wilddet3d"]), fetch(ck["cotracker"]); offline_ckpt = fetch(ck["cotracker_offline"])

    from libero.libero.envs import OffScreenRenderEnv
    bddls = sorted(glob.glob(os.path.join(bddl_dir, "*.bddl")))
    sel = [int(x) for x in args.tasks.split(",")] if args.tasks else range(len(bddls))
    fb = None
    for ti in sel:
        bddl = bddls[ti]; name = os.path.splitext(os.path.basename(bddl))[0]
        instruction = " ".join(re.search(r"\(:language\s+([^)]*)\)", open(bddl).read()).group(1).split())
        hdf5 = os.path.join(demo_dir, f"{name}_demo.hdf5")
        out_dir = os.path.join(cfg["train"]["cache_dir"], f"libero_{args.suite}__{name}"); os.makedirs(out_dir, exist_ok=True)
        print(f"[task {ti}] {name}: {instruction!r}\n  demos {hdf5}\n  cache {out_dir}", flush=True)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res, camera_depths=True)
        env.seed(0); env.reset(); sim = env.env.sim
        if fb is None:
            fb = FrameBuilder(sim, dict(pc, span_head=ck.get("span_head")), wd_ckpt, tracker_ckpt)
        fb.set_task(instruction)
        with h5py.File(hdf5, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
            if args.max_demos:
                demos = demos[:args.max_demos]
            for di, dk in enumerate(demos):
                outp = os.path.join(out_dir, f"{dk}.npz")
                if os.path.exists(outp):
                    print(f"  {dk}: exists, skip", flush=True); continue
                g = f[f"data/{dk}"]
                states = g["states"][:]; actions = g["actions"][:].astype(np.float32)
                joints = g["obs/joint_states"][:].astype(np.float32); grip = g["obs/gripper_states"][:].astype(np.float32)
                state8 = np.concatenate([joints, np.clip(grip[:, :1] / 0.04, 0.0, 1.0)], 1).astype(np.float32)
                idx = np.nonzero(np.ones(len(states), bool) if args.no_clean else noop_mask(actions))[0]; T = len(idx)
                t0 = time.time()
                # 1. frame-0 grounding + offline tracking of the kept frames
                fb.reset(sim); fb.offline_tracks = None
                frames = [np.flipud(env.set_init_state(states[t])["agentview_image"]).copy() for t in idx]
                o0 = fb.build(sim, env.set_init_state(states[idx[0]]), seed=int(idx[0]))
                cl0 = [tuple(map(float, o0["clicks"][k])) for k in fb.tk_slots]
                if cl0:
                    tr, vi = fb.track_offline(frames, cl0, offline_ckpt)
                    print(f"  {dk}: offline tracks T={len(tr)} K={tr.shape[1]} visible={vi.mean():.2f}", flush=True)
                del frames
                # 2. per-frame features, following the offline tracks
                fb.reset(sim)
                patches = np.zeros((T, G * G, 1024), np.float16)
                box2d = np.zeros((T, 4, 4), np.float32); xyz = np.zeros((T, 4, 3), np.float32); boxgeo = np.zeros((T, 4, 9), np.float32)
                pcd_pts = np.zeros((T, 4, n_pts, 3), np.float16); pcd_mask = np.zeros((T, 4, n_pts), bool)
                clicks = np.full((T, 4, 2), np.nan, np.float32)
                for fi, t in enumerate(idx):
                    o = fb.build(sim, env.set_init_state(states[t]), seed=int(t))
                    patches[fi] = o["patches"]; box2d[fi] = o["box2d"]; xyz[fi] = o["xyz"]; boxgeo[fi] = o["boxgeo"]
                    pcd_pts[fi] = o["pcd_pts"]; pcd_mask[fi] = o["pcd_mask"]; clicks[fi] = o["clicks"]
                    if fi % 50 == 0:
                        print(f"  {dk}: frame {fi}/{T} ({(time.time() - t0) / max(fi, 1):.2f} s/frame)", flush=True)
                tmp = outp + ".tmp"
                with open(tmp, "wb") as fh: np.savez_compressed(fh, state=state8[idx], action=actions[idx], patches=patches, box2d=box2d, xyz=xyz,
                                    boxgeo=boxgeo, pcd_pts=pcd_pts, pcd_mask=pcd_mask, clicks=clicks,
                                    image_wh=np.array([res, res], np.float32), instruction=instruction, kept=idx.astype(np.int32))
                os.replace(tmp, outp)
                fill = [round(float((np.abs(xyz[:, k]).sum(-1) > 0).mean()), 2) for k in range(4)]
                print(f"  [{di + 1}/{len(demos)}] {dk}: {T}/{len(states)} frames kept, {time.time() - t0:.0f}s, slot fill {dict(zip(SLOTS, fill))}", flush=True)
        env.close()
    print("[precompute] done", flush=True)


if __name__ == "__main__":
    main()
