"""Flow-matching training of the GAM policy.

    python -m GAM.train --config configs/my_task.yaml           # on the precomputed cache (train.cache_dir, local or hf://)
    python -m GAM.train --config configs/my_task.yaml --e2e     # end-to-end: raw episodes, frozen perception run on the fly

Loss: v-prediction MSE with t ~ 0.999*Beta(1, 1.5), x_t = (1-t) x0 + t x1, dt = 0.
EMA (power 0.75) weights are what gets saved. Checkpoints are written in the release
format, directly loadable by GAMPolicy.from_checkpoint / eval scripts.
"""
from __future__ import annotations

import argparse
import resource  # thousands of mmap'd cache files per rank -> raise the fd soft limit to the hard limit
try:
    _h = resource.getrlimit(resource.RLIMIT_NOFILE)[1]; resource.setrlimit(resource.RLIMIT_NOFILE, (_h, _h))
except Exception as _e:
    print("[fdlimit] failed:", _e)
import contextlib
import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GAM.config import load_config
from GAM.data.cache_dataset import CacheDataset, fit_limits_normalizer
from GAM.model.policy import GAMPolicy


def build_policy(cfg, ds) -> tuple:
    t = cfg["train"]
    S = len(cfg["data"]["slots"])
    n_cams = int(cfg["perception"].get("n_cams", 1))   # 2 = agentview + wrist camera, tokens concatenated
    n_patch = int(cfg["perception"].get("patch_grid", 16)) ** 2 * n_cams
    if hasattr(ds, "eps") and isinstance(ds.eps[0], dict):
        assert ds.eps[0]["patches"].shape[1] == n_patch, \
            f"cache patches {ds.eps[0]['patches'].shape[1]} != patch_grid^2*n_cams {n_patch}, re-run precompute"
    pp = int(t.get("patch_pool", 0))
    if pp: n_patch = pp * pp   # a 16x16 cache pooled to pxp before the model
    tps = int(t.get("tokens_per_slot", 8))
    lang_mode = str(t.get("lang_mode", "none"))    # none | film (sentence embedding modulates every layer's adaLN)
    assert lang_mode in ("none", "film"), f"train.lang_mode must be none or film, got {lang_mode!r}"
    config = dict(
        action_dim=int(ds.action_dim), horizon=int(t["horizon"]),
        n_action_steps=int(t["n_action_steps"]), num_inference_steps=10,
        n_emb=int(t.get("n_emb", 576)), n_layer=int(t.get("n_layer", 12)),
        n_head=int(t.get("n_head", 16)),
        stream_split=[n_patch, S * tps, 1],
        encoder=dict(n_patch=n_patch, patch_dim=1024, n_slots=S, tokens_per_slot=tps,
                     fourier_L=8, pcd_dim=int(t.get("pcd_dim", 512)),
                     state_dim=int(getattr(ds, "state_dim", ds.action_dim)) * int(t["state_hist"]),
                     fuse_heads=int(t.get("fuse_heads", 16)), fuse_layers=int(t.get("fuse_layers", 1)),   # number of fusion layers
                     state_dropout=float(t.get("state_dropout", 0.0)),
                     det_role_cam=bool(t.get("det_role_cam", True)),
                     det_mode=str(t.get("det_mode", "linear")), det_group=str(t.get("det_group", "fps"))),   # det_group: fps | octant; det_mode group = point-group tokens; det_role_cam False = no pick/place or camera embedding
        image_channel_swap=bool(cfg["perception"].get("image_channel_swap", False)),
        image_wh=list(ds.image_wh), train_image_wh=list(ds.image_wh),
        attn_dropout=float(t.get("attn_dropout", 0.0)),
        lang_film=(lang_mode == "film"), lang_dim=768,
        meta=dict(state_hist=int(t["state_hist"]), n_points=int(cfg["perception"].get("n_points", 512)),
                  lang_mode=lang_mode, tasks=list(t.get("tasks") or []), patch_pool=int(t.get("patch_pool", 0)), det_group=str(t.get("det_group", "fps")),
                  action=(f"absolute joint targets, {ds.action_dim}-d" if ds.action_mode == "abs"
                          else f"delta joint targets (+state[:7]), gripper abs, {ds.action_dim}-d"),
                  action_mode=ds.action_mode, pcd_ee_relative=bool(ds.pcd_ee_relative),
                  pcd_ref=ds.pcd_ref, n_cams=n_cams, patch_grid=int(cfg["perception"].get("patch_grid", 36)),
                  slots=cfg["data"]["slots"]),
    )
    policy = GAMPolicy(**{k: v for k, v in config.items() if k != "meta"})
    policy.meta = config["meta"]
    return policy, config


def optim_groups(policy, wd):
    """Reference-exact grouping: trunk Linear/MultiheadAttention weights decay; biases,
    LayerNorm/Embedding/RmsNorm weights and bare Parameters (pos embeddings, null tokens)
    do not; the observation encoder is one group WITH decay (as in the original recipe)."""
    import torch.nn as nn
    from timm.models.vision_transformer import RmsNorm
    decay, no_decay = [], []
    seen = set()
    for m in policy.model.modules():
        for pn, p in m.named_parameters(recurse=False):
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            if pn.endswith("bias") or pn.startswith("bias"):
                no_decay.append(p)
            elif isinstance(m, (nn.Linear, nn.MultiheadAttention)):
                decay.append(p)
            elif isinstance(m, (nn.LayerNorm, nn.Embedding, RmsNorm)):
                no_decay.append(p)
            else:                                    # bare Parameters: pos_emb, cond pos embed
                no_decay.append(p)
    enc = [p for p in policy.encoder.parameters() if p.requires_grad]
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": enc, "weight_decay": wd}]


def ema_decay(step, power=0.75, max_value=0.9999):
    if step <= 0:
        return 0.0
    return min(max_value, 1 - (1 + step) ** -power)


class _LossModule(torch.nn.Module):
    """Wraps encoder+trunk+flow-matching loss in one forward so DDP sees a single module."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, act):
        policy = self.policy
        B = act.shape[0]
        cond = policy.encoder(policy.normalizer.normalize(obs))
        x1 = policy.normalizer._affine("action", act, True)
        tt = 0.999 * torch.distributions.Beta(1.0, 1.5).sample((B,)).to(act.device)
        x0 = torch.randn_like(x1)
        x_t = (1 - tt.view(-1, 1, 1)) * x0 + tt.view(-1, 1, 1) * x1
        lang = obs.get("lang") if getattr(policy, "lang_film", False) else None
        v = policy.model(x_t, tt, torch.zeros_like(tt), cond, lang)
        return F.mse_loss(v.float(), (x1 - x0).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="override train.max_steps")
    ap.add_argument("--batch-size", type=int, default=0, help="override train.batch_size (per GPU)")
    ap.add_argument("--grad-accum", type=int, default=0, help="override train.grad_accum (micro-batches per optimiser step)")
    ap.add_argument("--e2e", action="store_true", help="train from the raw episodes (data.episodes_glob); perception computed on the fly, no cache")
    ap.add_argument("--resume", default="", help="latest.pt (EMA weights) to warm-start from")
    ap.add_argument("--resume-step", type=int, default=0, help="global step that ckpt was saved at")
    ap.add_argument("--rewarmup", type=int, default=200, help="linear lr re-warmup steps after resume (fresh AdamW moments)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = cfg["train"]
    if args.batch_size:
        t["batch_size"] = args.batch_size
    if args.grad_accum:
        t["grad_accum"] = args.grad_accum
    # DDP: launched via torchrun (env RANK/WORLD_SIZE); batch_size is PER-RANK.
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        import torch.distributed as dist
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    torch.manual_seed(int(t.get("seed", 42)) + rank)
    np.random.seed(int(t.get("seed", 42)) + rank)
    out_dir = t["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    S = len(cfg["data"]["slots"])
    featurizer = None; val_dl = None
    if args.e2e:
        if cfg.get("benchmark") == "libero_pro":          # demonstrations replayed in the simulator (benchmarks/libero_pro)
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "libero_pro"))
            from e2e_dataset import LiberoE2EDataset
            ds = LiberoE2EDataset(cfg, horizon=int(t["horizon"]), state_hist=int(t["state_hist"]))
        else:                                               # HDF5 episodes read by GAM.data.episodes
            from GAM.data.e2e_dataset import E2EDataset
            from GAM.config import perception_settings, build_extractor
            from GAM.precompute import build_arm_mask_fn, FrameFeaturizer
            assert t.get("action_mode", "abs") == "abs", "--e2e supports absolute joint actions"
            assert str(t.get("lang_mode", "none")) == "none", "--e2e does not compute the sentence embedding (train.lang_mode: film needs the cache path)"
            ds = E2EDataset(cfg, horizon=int(t["horizon"]), state_hist=int(t["state_hist"]))
            featurizer = FrameFeaturizer(cfg, build_extractor(perception_settings(cfg)), build_arm_mask_fn(cfg))
        if rank == 0:
            n_ep = len(getattr(ds, "eps", None) or getattr(ds, "episodes", []))
            print(f"[train] end-to-end: {n_ep} episodes / {len(ds)} frames, perception on the fly", flush=True)
    else:
        from GAM.hub import fetch
        t["cache_dir"] = fetch(t["cache_dir"].rstrip("/") + "/")   # hf://... caches are downloaded on first use
        _pat = t.get("cache_glob", "*.npz")
        all_names = sorted(os.path.splitext(os.path.relpath(p, t["cache_dir"]))[0]
                           for p in __import__("glob").glob(os.path.join(t["cache_dir"], _pat)))
        if t.get("tasks"):   # train only on these task sub-directories
            _ts = set(t["tasks"]); all_names = [n for n in all_names if n.split("/")[0] in _ts]
            assert all_names, "train.tasks matched nothing"
        n_val = int(cfg["data"].get("val_episodes", 0))
        train_names = all_names[:-n_val] if n_val else all_names
        ds = CacheDataset(t["cache_dir"], train_names, horizon=int(t["horizon"]), n_slots=S,
                          state_hist=int(t["state_hist"]), action_mode=t.get("action_mode", "abs"),
                          pcd_ee_relative=bool(t.get("pcd_ee_relative", False)),
                          pcd_ref=str(t.get("pcd_ref", "none")), pattern=_pat, patch_pool=int(t.get("patch_pool", 0)))
        if rank == 0:
            print(f"[train] {len(ds.eps)} episodes / {len(ds)} windows "
                  f"(val held out: {all_names[len(train_names):]}) world={world}", flush=True)
        val_names = all_names[len(train_names):]
        if n_val and rank == 0:   # held-out validation loss (over-fitting check), computed on rank 0 at every checkpoint
            val_ds = CacheDataset(t["cache_dir"], val_names, horizon=int(t["horizon"]), n_slots=S,
                                  state_hist=int(t["state_hist"]), action_mode=t.get("action_mode", "abs"),
                                  pcd_ee_relative=bool(t.get("pcd_ee_relative", False)),
                                  pcd_ref=str(t.get("pcd_ref", "none")), pattern=_pat)
            val_dl = DataLoader(val_ds, batch_size=int(t["batch_size"]), shuffle=False,
                                num_workers=4, pin_memory=False, drop_last=False)
            print(f"[train] val: {len(val_ds.eps)} episodes / {len(val_ds)} windows", flush=True)
    sampler = None
    if world > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True,
                                     seed=int(t.get("seed", 42)), drop_last=True)
    dl = DataLoader(ds, batch_size=int(t["batch_size"]), shuffle=(sampler is None), sampler=sampler,
                    num_workers=0 if args.e2e else int(t.get("num_workers", 8)), pin_memory=False, drop_last=True,
                    collate_fn=ds.collate if args.e2e else None)

    policy, config = build_policy(cfg, ds)
    scale, offset = fit_limits_normalizer(ds.all_actions())
    policy.normalizer.add("action", scale, offset)
    if args.resume:                                     # warm start from a previous run
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        mis, unexp = policy.load_state_dict(ck["state_dict"], strict=False)
        mis = [m for m in mis if not m.startswith("normalizer.")]
        assert not mis and not unexp, (mis, unexp)
        if rank == 0:
            print(f"[train] RESUME weights from {args.resume} @ step {args.resume_step}", flush=True)
    policy = policy.cuda().train()
    net = _LossModule(policy)
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
            find_unused_parameters=True,   # null tokens are used data-dependently
            gradient_as_bucket_view=True)
    if rank == 0:
        print(f"[train] params {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M", flush=True)

    max_steps = args.max_steps or int(t["max_steps"])
    warmup = int(t.get("warmup_steps", 1000))
    opt = torch.optim.AdamW(optim_groups(policy, float(t.get("weight_decay", 1e-3))),
                            lr=float(t["lr"]), betas=(0.9, 0.95), eps=1e-8)
    _r0 = int(args.resume_step) if args.resume else 0
    def _lr_lambda(s):
        s = s + _r0                                     # scheduler restarts at 0 -> shift to global step
        base = s / max(warmup, 1) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(max_steps - warmup, 1)))
        if _r0 and args.rewarmup > 0:
            base *= min(1.0, (s - _r0 + 1) / args.rewarmup)
        return base
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    # EMA on CPU (rank 0): frees ~1.2GB of VRAM; per-step D2H copy of params (~60ms)
    ema_dev = str(t.get("ema_device", "cpu"))   # EMA on the CPU by default; large models can keep it on the GPU (faster, more memory)
    ema = copy.deepcopy(policy).to(ema_dev).eval().requires_grad_(False) if rank == 0 else None
    xyz_drop = float(t.get("det_xyz_dropout", 0.0))
    patch_drop = float(t.get("patch_dropout", 0.0))   # zero image tokens at random (per token, per sample)
    patch_drop_all = float(t.get("patch_drop_all", 0.0))   # zero all image tokens of a sample with this probability (forces the policy onto the object geometry)
    slot_drop = float(t.get("slot_dropout", 0.0))   # empty a whole (sample, slot) at random, leaks the task phase when slots drop out systematically; off by default

    step, epoch, t0, run_loss = 0, 0, time.time(), 0.0
    accum = int(t.get("grad_accum", 1)); micro = 0
    if rank == 0 and accum > 1:
        print(f"[train] grad_accum={accum}: effective batch {int(t['batch_size']) * accum * world}", flush=True)
    if args.resume:
        step = int(args.resume_step); epoch = step // max(len(dl), 1)   # different shuffle than a cold start
    best_val = float("inf")
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in dl:
            if args.e2e:
                with torch.no_grad():
                    obs, act = ds.featurize_batch(batch, featurizer)
            else:
                obs = {k: v.cuda(non_blocking=True) for k, v in batch["obs"].items()}
                act = batch["action"].cuda(non_blocking=True)
            B = act.shape[0]
            if patch_drop > 0:
                pt = obs["patch_tokens"]
                keep_p = (torch.rand(pt.shape[0], pt.shape[1], 1, device=pt.device) >= patch_drop).to(pt.dtype)
                obs["patch_tokens"] = pt * keep_p
            if patch_drop_all > 0:
                pt = obs["patch_tokens"]
                keep_s = (torch.rand(pt.shape[0], 1, 1, device=pt.device) >= patch_drop_all).to(pt.dtype)
                obs["patch_tokens"] = pt * keep_s
            if slot_drop > 0:
                km = (torch.rand(B, obs["xyz"].shape[1], 1, device=act.device) >= slot_drop)
                kf = km.float()
                obs["xyz"] = obs["xyz"] * kf; obs["boxgeo"] = obs["boxgeo"] * kf; obs["box2d"] = obs["box2d"] * kf
                obs["pcd_mask"] = obs["pcd_mask"] & km; obs["pcd_pts"] = obs["pcd_pts"] * kf[..., None]
            if xyz_drop > 0:                                # object 3D-centre dropout per sample
                keep = (torch.rand(B, 1, 1, device=act.device) >= xyz_drop).float()
                obs["xyz"] = obs["xyz"] * keep
            micro += 1   # gradient accumulation: grad_accum micro-batches = 1 optimiser step
            sync_now = (micro % accum == 0)
            ctx = net.no_sync() if (world > 1 and not sync_now) else contextlib.nullcontext()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = net(obs, act)
                (loss / accum).backward()
            if not sync_now:
                continue
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            # EMA (rank 0 only; DDP keeps replicas identical)
            if ema is not None:
                d = ema_decay(step)
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), policy.parameters()):
                        pe.mul_(d).add_(p.detach().to(pe.device), alpha=1 - d)
                    for be, b in zip(ema.buffers(), policy.buffers()):
                        be.copy_(b.to(be.device))
            run_loss += loss.item()
            if step % 50 == 0 and rank == 0:
                print(f"step {step}/{max_steps}  loss {run_loss / 50:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {(time.time() - t0) / max(step - _r0, 1):.2f}s/it  "
                      f"gpu-mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                      flush=True)
            if step % 50 == 0:
                run_loss = 0.0
            if rank == 0 and (step % int(t.get("save_every", 1000)) == 0 or step == max_steps):
                ema.meta = policy.meta
                def _save_retry(path, tries=4):   # a transient I/O error must not kill the run
                    for k in range(tries):
                        try:
                            ema.save_checkpoint(path, config); return True
                        except Exception as _e:  # noqa: BLE001
                            print(f"[train] save {os.path.basename(path)} failed ({_e}), retry {k + 1}/{tries}", flush=True)
                            time.sleep(15 * (k + 1))
                    print(f"[train] WARNING: give up saving {path}", flush=True); return False
                if _save_retry(os.path.join(out_dir, "latest.pt")):
                    print(f"[train] saved latest.pt @ step {step}", flush=True)
                if step % (int(t.get("snapshot_every_saves", 5)) * int(t.get("save_every", 1000))) == 0:
                    _save_retry(os.path.join(out_dir, f"step_{step}.pt"))
                if val_dl is not None:   # held-out loss (eval mode: no dropout)
                    policy.eval(); vl, nb = 0.0, 0
                    _vnet = net.module if hasattr(net, "module") else net
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for vb in val_dl:
                            vobs = {k: v.cuda(non_blocking=True) for k, v in vb["obs"].items()}
                            vl += float(_vnet(vobs, vb["action"].cuda(non_blocking=True))); nb += 1
                            if nb >= int(t.get("val_batches", 40)):
                                break
                    policy.train()
                    vmean = vl / max(nb, 1)
                    if vmean < best_val:   # best.pt = the EMA weights with the lowest validation loss
                        best_val = vmean
                        ema.save_checkpoint(os.path.join(out_dir, "best.pt"), config)
                        with open(os.path.join(out_dir, "best.txt"), "w") as f:
                            f.write(f"step {step} val_loss {vmean:.4f}\n")
                    print(f"[val] step {step}  val_loss {vmean:.4f}  (train ~{loss.item():.4f})"
                          f"  best {best_val:.4f}{' *' if vmean == best_val else ''}", flush=True)
            if step >= max_steps:
                break
    if rank == 0:
        print("[train] done")


if __name__ == "__main__":
    main()
