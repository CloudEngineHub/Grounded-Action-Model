"""Window dataset over the precomputed cache (one .npz per episode)."""
from __future__ import annotations

import glob
import os
from typing import List

import numpy as np
import torch
from GAM.data.patchpool import pool_patches as _pool
from torch.utils.data import Dataset


class CacheDataset(Dataset):
    """Each sample: observation at frame t (state history included) + the next `horizon`
    actions (clamped at the episode end, repeating the last action)."""

    def __init__(self, cache_dir: str, episodes: List[str] | None = None, horizon: int = 16,
                 state_hist: int = 2, action_mode: str = "abs", pcd_ee_relative: bool = False,
                 pcd_ref: str = "none", pattern: str = "*.npz", n_slots: int = 0, patch_pool: int = 0):
        # n_slots > 0: only the first n_slots slots of the cache are used
        # pcd_ref: "none" | "ee" (object geometry relative to the end effector) | "pick" (reference = the pick object's centre:
        #   the pick cloud becomes object-centred, the other slots become offsets from it, and the pick slot's xyz = centre - TCP)
        if pcd_ee_relative and pcd_ref == "none":
            pcd_ref = "ee"
        self.pcd_ref = pcd_ref
        pcd_ee_relative = pcd_ref != "none"
        # pcd_ee_relative: object xyz / point clouds minus the current TCP position (base frame, FK from state[:7]) every frame;
        # boxgeo and actions unchanged, removes the absolute-position fingerprint
        self.pcd_ee_relative = pcd_ee_relative
        # action_mode: "abs" = absolute joint targets; "delta_joint" = arm joints relative to the current state (gripper stays absolute)
        self.action_mode = action_mode
        paths = sorted(glob.glob(os.path.join(cache_dir, pattern)))   # the pattern may include sub-directories (LIBERO: */*.npz)
        _name = lambda p: os.path.splitext(os.path.relpath(p, cache_dir))[0]
        if episodes is not None:
            paths = [p for p in paths if _name(p) in episodes]
        assert paths, f"no cache episodes in {cache_dir}"
        self.eps = [dict(np.load(p, allow_pickle=True)) for p in paths]
        for ep, p in zip(self.eps, paths):   # large patch caches live in a separate .patches.npy, memory-mapped
            pn = os.path.splitext(p)[0] + ".patches.npy"
            if os.path.exists(pn):
                ep["patches"] = np.load(pn, mmap_mode="r")
        self.patch_pool = int(patch_pool)   # > 0: pool the cached masked patch grid to pxp (GAM.data.patchpool.pool_patches)
        if n_slots and n_slots > 0:
            for ep in self.eps:
                for k in ("box2d", "xyz", "boxgeo", "pcd_pts", "pcd_mask", "clicks"):
                    if k in ep and ep[k].ndim >= 2 and ep[k].shape[1] > n_slots:
                        ep[k] = np.ascontiguousarray(ep[k][:, :n_slots])
        self.names = [_name(p) for p in paths]
        self.horizon = horizon
        self.state_hist = state_hist
        self.index = [(e, t) for e, ep in enumerate(self.eps) for t in range(len(ep["state"]))]
        if pcd_ee_relative:
            fk = ee_fk()
            for ep in self.eps:
                ep["ee_pos"] = np.stack([fk(st[:7]) for st in ep["state"]], 0).astype(np.float32)
        self.action_dim = self.eps[0]["action"].shape[-1]
        self.state_dim = self.eps[0]["state"].shape[-1]       # LIBERO: action 7 (OSC) != state 8
        self.image_wh = tuple(float(v) for v in self.eps[0]["image_wh"])

    def __len__(self):
        return len(self.index)

    def _act_window(self, ep, t):
        T = len(ep["state"])
        idx = np.clip(np.arange(t, t + self.horizon), 0, T - 1)
        a = ep["action"][idx].astype(np.float32).copy()
        if self.action_mode == "delta_joint":
            a[:, :7] -= ep["state"][t][:7].astype(np.float32)
        return a

    def all_actions(self) -> np.ndarray:
        if self.action_mode == "delta_joint":   # normalisation statistics on the relative actions too (one window every 8 frames is enough)
            return np.concatenate([self._act_window(ep, t) for ep in self.eps
                                   for t in range(0, len(ep["state"]), 8)], 0)
        return np.concatenate([ep["action"] for ep in self.eps], 0)

    def obs_at(self, e: int, t: int) -> dict:
        ep = self.eps[e]
        hist = [ep["state"][max(t - k, 0)] for k in range(self.state_hist - 1, -1, -1)]
        xyz, pts = ep["xyz"][t], ep["pcd_pts"][t].astype(np.float32)
        if self.pcd_ee_relative:
            xyz, pts = relativize(xyz, pts, ep["pcd_mask"][t], ep["ee_pos"][t], ref=self.pcd_ref)
        obs = {
            "patch_tokens": (_pool(torch.from_numpy(ep["patches"][t].astype(np.float32)), int(round(ep["patches"].shape[1] ** 0.5)), self.patch_pool) if self.patch_pool else torch.from_numpy(ep["patches"][t].astype(np.float32))),
            "box2d": torch.from_numpy(ep["box2d"][t]),
            "xyz": torch.from_numpy(np.asarray(xyz)),
            "boxgeo": torch.from_numpy(ep["boxgeo"][t]),
            "pcd_pts": torch.from_numpy(pts),
            "pcd_mask": torch.from_numpy(ep["pcd_mask"][t]),
            "state": torch.from_numpy(np.concatenate(hist).astype(np.float32)),
            "lang": torch.from_numpy(np.asarray(ep["lang"], np.float32)) if "lang" in ep
                    else torch.zeros(768),
        }
        return obs

    def __getitem__(self, i: int):
        e, t = self.index[i]
        return {"obs": self.obs_at(e, t),
                "action": torch.from_numpy(self._act_window(self.eps[e], t))}


def relativize(xyz, pts, mask, ee, ref="ee"):
    """xyz (S,3), pts (S,n,3), mask (S,n). ref="ee": valid objects / points minus the TCP position.
    ref="pick": reference = the centre c of the pick slot (0): every valid point and the other slots' xyz minus c, the pick slot's xyz := c - TCP;
    falls back to ref="ee" when the pick slot is empty. Empty slots (xyz all zero) and invalid points stay 0."""
    xyz = np.asarray(xyz, np.float32).copy(); pts = np.asarray(pts, np.float32).copy()
    ee = np.asarray(ee, np.float32); mask = np.asarray(mask, bool)
    valid = np.abs(xyz).sum(-1) > 0
    if ref == "pick" and valid[0]:
        c = xyz[0].copy()
        pts[mask] -= c
        xyz[valid] -= c
        xyz[0] = c - ee
        return xyz, pts
    xyz[valid] -= ee
    pts[mask] -= ee
    return xyz, pts


def ee_fk():
    """q7 -> TCP position in the robot base frame (Franka Panda)."""
    from GAM.data.panda_fk import tcp_position
    return tcp_position


def fit_limits_normalizer(actions: np.ndarray, out_max=1.0, out_min=-1.0, eps=1e-4):
    """x_norm = x * scale + offset mapping per-dim [min,max] -> [-1,1] (constant dims -> mid)."""
    a = torch.from_numpy(np.asarray(actions, np.float32))
    lo, hi = a.min(0).values, a.max(0).values
    rng = hi - lo
    ignore = rng < eps
    rng = torch.where(ignore, torch.full_like(rng, out_max - out_min), rng)
    scale = (out_max - out_min) / rng
    offset = out_min - scale * lo
    offset[ignore] = (out_max + out_min) / 2 - lo[ignore]
    return scale, offset
