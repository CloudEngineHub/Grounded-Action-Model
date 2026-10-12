"""Object tracks from language, the one offline step of end-to-end training.

    python -m GAM.tools.track --config configs/lerobot_bimanual.yaml [--ep-start 0 --ep-end 999]

Per episode (HDF5 or LeRobot, whatever data.episodes_glob / data.source says):
  1. instruction -> object phrases:  data.slots: auto  = the T5 span head (checkpoints.span_head)
                                      data.slots: [...]  = the names you give
  2. frame 0: text-prompted WildDet3D detection of every phrase -> one start point per object
  3. the whole episode: offline CoTracker (checkpoints.cotracker_offline) follows the points;
     a point that is lost keeps its last visible position
Output: <data.clicks.tracks_dir>/<episode>.npz with clicks (T,S,2) px (NaN = never found), seg_id (T,) zeros,
prompts, instruction, read back through data.clicks: {source: tracks, tracks_dir: ...}, so GAM.precompute
and `GAM.train --e2e` then run the detector with per-frame point prompts (perception.grounding: click).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from GAM.config import load_config, perception_settings, build_extractor
from GAM.data.episodes import list_episodes, load_episode
from GAM.hub import fetch


def phrases_for(cfg, instruction, n_slots, head=None):
    """Slot prompts of one episode: the span head's phrases (slots: auto) or the configured names."""
    slots = cfg["data"]["slots"]
    if not cfg["data"].get("slots_auto"):
        return [s for s in slots if s != "__none__"][:n_slots]
    assert head is not None, "data.slots: auto needs checkpoints.span_head"
    from GAM.text.phrases import manipulation_objects
    return manipulation_objects(instruction, head.extract(instruction))[:n_slots]   # object names, landmarks dropped


def track_offline(frames, points, ckpt, device="cuda"):
    """frames [(H,W,3) uint8] x T, points [(x,y)] -> (T,K,2) px, (T,K) visible (CoTracker3 offline)."""
    import torch
    from cotracker.predictor import CoTrackerPredictor
    if not hasattr(track_offline, "_p") or track_offline._ckpt != ckpt:
        track_offline._p = CoTrackerPredictor(checkpoint=ckpt, offline=True, window_len=60).to(device).eval(); track_offline._ckpt = ckpt
    vid = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()[None].to(device)
    q = torch.tensor([[[0.0, float(x), float(y)] for x, y in points]], device=device)
    with torch.no_grad():
        tr, vi = track_offline._p(vid, queries=q)
    return tr[0].float().cpu().numpy(), vi[0].float().cpu().numpy() > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ep-start", type=int, default=0); ap.add_argument("--ep-end", type=int, default=10**9)
    args = ap.parse_args()
    cfg = load_config(args.config); d, ck, tr = cfg["data"], cfg["checkpoints"], cfg["train"]
    n_slots = len(d["slots"])
    out_dir = (d.get("clicks") or {}).get("tracks_dir") or tr.get("tracks_dir"); assert out_dir, "data.clicks.tracks_dir is not set"
    os.makedirs(out_dir, exist_ok=True)
    pc = perception_settings(cfg); ext = build_extractor(pc)
    text_min = float(pc.get("text_score_min", 0.3))
    head = None
    if d.get("slots_auto"):
        from GAM.text.segmenter import T5ObjectSegmenter
        head = T5ObjectSegmenter.from_checkpoint(fetch(ck["span_head"]))
    tracker_ckpt = fetch(ck["cotracker_offline"])
    eps = list_episodes(dict(cfg, data=dict(d, clicks={"source": "none"})))[args.ep_start:args.ep_end]
    for i, path in enumerate(eps):
        name = f"episode_{path.split('#')[-1]}" if path.startswith("lerobot://") else os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, name + ".npz")
        if os.path.exists(out):
            print(f"[{i + 1}/{len(eps)}] {name}: exists, skip"); continue
        ep = load_episode(path, dict(cfg, data=dict(d, clicks={"source": "none"})))
        prompts = phrases_for(cfg, ep.instruction, n_slots, head)
        T = ep.T; clicks = np.full((T, n_slots, 2), np.nan, np.float32)
        rgb0 = ep.rgb(0); H, W = rgb0.shape[:2]
        found = []
        if prompts:
            r = ext(rgb0, np.zeros((H, W), np.float32), ep.K[0].astype(np.float64), ep.E[0].astype(np.float64), [], texts=prompts)
            for k, det in enumerate(r.detections):
                if det.score >= text_min and det.box2d[2] > det.box2d[0]:
                    found.append((k, float((det.box2d[0] + det.box2d[2]) / 2), float((det.box2d[1] + det.box2d[3]) / 2)))
        if found:
            frames = [ep.rgb(t) for t in range(T)]
            pts, vis = track_offline(frames, [(u, v) for _, u, v in found], tracker_ckpt)
            del frames
            for j, (k, _, _) in enumerate(found):
                last = None
                for t in range(T):
                    ok = vis[t, j] and 0 <= pts[t, j, 0] < W and 0 <= pts[t, j, 1] < H
                    if ok:
                        last = pts[t, j]
                    if last is not None:
                        clicks[t, k] = last
        with open(out + ".tmp", "wb") as fh:                   # never a *.npz name while incomplete
            np.savez_compressed(fh, clicks=clicks, seg_id=np.zeros(T, np.int32),
                                prompts=np.array(prompts + [""] * (n_slots - len(prompts))), instruction=np.str_(ep.instruction))
        os.replace(out + ".tmp", out); ep.close()
        vis_rate = {prompts[k]: round(float(np.isfinite(clicks[:, k, 0]).mean()), 2) for k, _, _ in found}
        print(f"[{i + 1}/{len(eps)}] {name}: {ep.instruction!r} -> {prompts}; tracked {vis_rate}", flush=True)
    print(f"[track] done: {len(eps)} episodes -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
