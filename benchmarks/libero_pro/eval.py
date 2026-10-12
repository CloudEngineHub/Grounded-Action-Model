"""Closed-loop evaluation of a GAM policy on LIBERO-PRO.

    python benchmarks/libero_pro/eval.py --config configs/libero_pro.yaml --suite object --variant language
        [--tasks 0,3] [--episodes 50] [--ckpt my_policy.pt] [--chunking rtc] [--no-video]

The config fixes every perception setting (they must match the training cache); the policy is
`checkpoints.policy` of the config, the released checkpoint, downloaded from the Hub, unless
--ckpt is given. Per-frame observations come from frame_builder.FrameBuilder, the same code that
builds the training cache, from the agentview RGB-D, the camera calibration, the joint state and
the instruction only.

Results: <out>/results.jsonl (one line per episode; already finished episodes are skipped on
re-runs) and, with save_video, one mp4 per episode with the per-slot 3D boxes the policy saw drawn on the camera image.
"""
import argparse, glob, json, math, os, re, sys, time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.6")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from GAM.config import load_yaml
from GAM.hub import fetch
from frame_builder import FrameBuilder, SLOTS
from GAM.perception.draw import draw_slots
from GAM.eval.chunker import ActionChunker, predict_with_prefix

SUITES = ["spatial", "object", "goal", "10"]
VARIANTS = ["language", "object", "swap", "task"]
STEP_BUDGET = {"10": 520}                      # official per-suite step limits above eval.max_steps


def suite_dir(suite, variant):
    return "libero_spatial_swap" if (suite, variant) == ("spatial", "swap") else f"libero_{suite}_temp_{variant}"


def task_dirs(cfg, name):
    """(bddl_dir, init_dir) of one task folder from paths.task_files (a local folder or hf://... with
    bddl_files/ and init_files/; the released set replaces LIBERO-PRO's incomplete initial states)."""
    root = fetch(cfg["paths"]["task_files"].rstrip("/") + "/")
    bddl_dir, init_dir = os.path.join(root, "bddl_files", name), os.path.join(root, "init_files", name)
    assert os.path.isdir(bddl_dir), f"{bddl_dir} not found (paths.task_files)"
    return bddl_dir, init_dir


def draw(frm, o, fb):
    """The policy's input on this frame: the per-slot 3D boxes (world centre + extents + rotation) projected with the camera."""
    short = lambda t: re.split(r"\b(?:between|on|in|next|from|beside|of|with|at|to|under)\b", t)[0].strip() if t else None   # "black bowl between the plate and ..." -> "black bowl"
    names = [short(fb.prompts[k]) if fb.prompts and fb.prompts[k] else None for k in range(4)]
    return draw_slots(frm, o["xyz"], o["boxgeo"], fb.K, fb.last_E, names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", choices=SUITES, required=True); ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--ckpt", default=None, help="policy checkpoint (local path or hf://...); default: checkpoints.policy")
    ap.add_argument("--tasks", default=None, help="comma-separated task indices (default: all)")
    ap.add_argument("--episodes", type=int, default=None, help="per task (default: eval.episodes_per_task)")
    ap.add_argument("--max-steps", type=int, default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--chunking", choices=["tempens", "rtc"], default=None, help="default: eval.chunking")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config); pc, ck, ev = cfg["perception"], cfg["checkpoints"], cfg["eval"]
    res = int(pc.get("image_size", 512))
    bddl_dir, init_dir = task_dirs(cfg, suite_dir(args.suite, args.variant))
    cell = f"{args.suite}/{args.variant}"
    out = args.out or os.path.join(ev["out_dir"], f"{args.suite}_{args.variant}"); os.makedirs(out, exist_ok=True)
    n_eps = args.episodes or int(ev["episodes_per_task"])
    max_steps = max(args.max_steps or int(ev["max_steps"]), STEP_BUDGET.get(args.suite, 0))
    save_video = not args.no_video and ev.get("save_video", True)
    mode = args.chunking or ev.get("chunking", "tempens")
    te, rtc = ev.get("tempens") or {}, ev.get("rtc") or {}
    shift = pc.get("click_shift") or {}

    from libero.libero.envs import OffScreenRenderEnv
    from GAM.model.policy import GAMPolicy
    ckpt = fetch(args.ckpt or ck["policy"]); wd_ckpt, tracker_ckpt = fetch(ck["wilddet3d"]), fetch(ck["cotracker"])
    policy = GAMPolicy.from_checkpoint(ckpt); dev = next(policy.parameters()).device
    meta = getattr(policy, "meta", {}) or {}; hist = int(meta.get("state_hist", 2))
    assert int(meta.get("n_cams", 1)) == 1, "this evaluation uses the agentview camera only"
    pcd_ref = meta.get("pcd_ref") or ("ee" if meta.get("pcd_ee_relative") else "none")
    uses_lang = bool(getattr(policy, "lang_film", False))
    if uses_lang:
        from GAM.text.language_clicks import sentence_embedding
    if pcd_ref != "none":
        from GAM.data.cache_dataset import relativize, ee_fk; tcp = ee_fk()
    chunker = ActionChunker(mode, k=float(te.get("k", 0.01)), query_every=int(te.get("query_every", 1)),
                            exec_steps=int(rtc.get("exec_steps", 8)), delay=int(rtc.get("delay", 1)))
    print(f"[eval] {cell}: {n_eps} episodes/task, max_steps={max_steps}, chunking={mode} "
          + (f"k={chunker.k}" if mode == "tempens" else f"exec_steps={chunker.exec_steps} delay={chunker.delay}"), flush=True)
    print(f"[eval] policy={ckpt}\n[eval] perception: grounding={pc['grounding']} detector_depth={pc['detector_depth']} "
          f"geometry_depth={pc['geometry_depth']} geometry_source={pc['geometry_source']} precision={pc.get('precision', 'fp32')} pcd_ref={pcd_ref}", flush=True)

    res_p = os.path.join(out, "results.jsonl"); done = set()
    if os.path.exists(res_p):
        for ln in open(res_p):
            try:
                r = json.loads(ln); done.add((r["task"], r["ep"])) if "success" in r else None
            except Exception:
                pass
    bddls = sorted(glob.glob(os.path.join(bddl_dir, "*.bddl")))
    sel = [int(x) for x in args.tasks.split(",")] if args.tasks else range(len(bddls))
    tot = suc = 0; fb = None
    for ti in sel:
        bddl = bddls[ti]; name = os.path.splitext(os.path.basename(bddl))[0]
        inits = glob.glob(os.path.join(init_dir, name + "*.pruned_init")) or glob.glob(os.path.join(init_dir, name + "*"))
        init_states = torch.load(inits[0], weights_only=False) if inits else None
        n_task = min(n_eps, len(init_states)) if init_states is not None else n_eps
        instruction = " ".join(re.search(r"\(:language\s+([^)]*)\)", open(bddl).read()).group(1).split())
        cs = (float(shift["px"][0]), float(shift["px"][1]), [int(v) for v in shift.get("slots", [0, 2])]) \
            if f"{cell}/{ti}" in (shift.get("tasks") or []) else None
        print(f"[task {ti}] {name}: {instruction!r}", flush=True)
        lang = torch.from_numpy(np.asarray(sentence_embedding(cfg, instruction), np.float32)).reshape(1, 768).to(dev) \
            if uses_lang else torch.zeros(1, 768, device=dev)
        try:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res, camera_depths=True)
            env.seed(0); env.reset()
        except Exception as e:
            open(res_p, "a").write(json.dumps({"task": name, "ti": ti, "ep": -1, "error": str(e)[:200]}) + "\n")
            print(f"[task {ti}] cannot build the environment: {e}", flush=True); continue
        errors = 0; first = True
        for ep in range(n_task):
            if (name, ep) in done:
                continue
            try:
                env.reset()
                if init_states is not None:
                    env.set_init_state(init_states[ep])
                for _ in range(5):
                    obs, _, _, _ = env.step(np.zeros(7))
                sim = env.env.sim
                if fb is None:
                    fb = FrameBuilder(sim, dict(pc, span_head=ck.get("span_head")), wd_ckpt, tracker_ckpt)
                if first:
                    fb.set_task(instruction, cs); first = False
                fb.reset(sim); chunker.reset()
                frames, st_hist, t0, steps, ok = [], [], time.time(), 0, False
                while steps < max_steps and not ok:
                    o = fb.build(sim, obs, seed=steps)
                    if save_video:
                        frames.append(draw(o["rgb"].copy(), o, fb))      # before the geometry is re-expressed relative to the gripper
                    q7 = np.asarray(obs["robot0_joint_pos"], np.float64)[:7]
                    if pcd_ref != "none":
                        o["xyz"], o["pcd_pts"] = relativize(o["xyz"], o["pcd_pts"], o["pcd_mask"], tcp(q7), ref=pcd_ref)
                    g01 = float(np.clip(obs["robot0_gripper_qpos"][0] / 0.04, 0.0, 1.0))
                    st_hist = (st_hist + [np.concatenate([q7.astype(np.float32), [g01]]).astype(np.float32)])[-hist:]
                    while len(st_hist) < hist:
                        st_hist.insert(0, st_hist[0])
                    t = lambda a: torch.as_tensor(np.asarray(a))[None].to(dev)
                    od = {"patch_tokens": t(o["patches"].astype(np.float32)), "box2d": t(o["box2d"]), "xyz": t(o["xyz"]),
                          "boxgeo": t(o["boxgeo"]), "pcd_pts": t(o["pcd_pts"].astype(np.float32)), "pcd_mask": t(o["pcd_mask"]),
                          "state": t(np.concatenate(st_hist).astype(np.float32)), "lang": lang}
                    a = chunker.act(lambda prefix: predict_with_prefix(policy, od, prefix, chunker.delay))
                    obs, _, ok, _ = env.step(np.clip(a.astype(np.float32), -1.0, 1.0)); steps += 1
            except Exception as e:
                errors += 1
                open(res_p, "a").write(json.dumps({"task": name, "ti": ti, "ep": ep, "error": str(e)[:200]}) + "\n")
                import traceback; traceback.print_exc()
                if errors >= 3:
                    print(f"[task {ti}] giving up after 3 consecutive errors", flush=True); break
                continue
            errors = 0; ok = bool(ok); tot += 1; suc += ok
            open(res_p, "a").write(json.dumps({"task": name, "ti": ti, "ep": ep, "success": ok, "steps": steps, "secs": round(time.time() - t0, 1)}) + "\n")
            if frames:
                try:
                    import imageio
                    imageio.mimwrite(os.path.join(out, f"t{ti:02d}_ep{ep:02d}_{'S' if ok else 'F'}.mp4"), frames[::2], fps=15, quality=6)
                except Exception as e:                      # a failed video must not lose the episode
                    print(f"  ep{ep}: video not saved ({type(e).__name__})", flush=True)
            print(f"  ep{ep}: {'success' if ok else 'fail'} steps={steps} ({time.time() - t0:.0f}s) running {suc}/{tot}", flush=True)
        env.close()
    print(f"[final] {cell}: {suc}/{tot} = {100.0 * suc / max(tot, 1):.1f}%", flush=True)


if __name__ == "__main__":
    main()
