"""End-to-end training data: raw episodes, features computed on the fly.

The frozen perception stack (WildDet3D, robot silhouette, object geometry) runs inside the training
loop through the same FrameFeaturizer as the cache precompute, so nothing is written to disk. The
cost is one perception forward per training sample (~0.6-1 s per frame on one GPU), which makes
this much slower than training on a cache; use it when the cache does not fit on disk or when the
perception settings are still being changed.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from GAM.data.episodes import list_episodes, load_episode


class E2EDataset(Dataset):
    """Each sample: (episode, frame) + the next `horizon` actions; the observation is built later by
    featurize_batch() on the GPU. Episodes stay open (one HDF5 handle each); use num_workers=0."""

    def __init__(self, cfg: dict, horizon: int, state_hist: int, names=None):
        paths = list_episodes(cfg)
        if names is not None:
            paths = [p for p in paths if any(p.endswith(n) for n in names)]
        self.cfg, self.horizon, self.state_hist = cfg, horizon, state_hist
        self.eps = [load_episode(p, cfg) for p in paths]
        self.index = [(e, t) for e, ep in enumerate(self.eps) for t in range(ep.T)]
        self.action_dim = self.eps[0].action.shape[-1]; self.state_dim = self.eps[0].state.shape[-1]
        H, W = self.eps[0].rgb(0).shape[:2]; self.image_wh = (float(W), float(H))
        self.pcd_ref = str(cfg.get("train", {}).get("pcd_ref", "none")); self.action_mode, self.pcd_ee_relative = "abs", self.pcd_ref != "none"
        if self.pcd_ee_relative:                                    # single-arm Franka: geometry relative to the TCP (Panda FK on state[:7])
            from GAM.data.cache_dataset import relativize, ee_fk
            self._relativize, self._tcp = relativize, ee_fk()

    def __len__(self):
        return len(self.index)

    def all_actions(self) -> np.ndarray:
        return np.concatenate([ep.action for ep in self.eps], 0)

    def __getitem__(self, i: int):
        e, t = self.index[i]; ep = self.eps[e]
        idx = np.clip(np.arange(t, t + self.horizon), 0, ep.T - 1)
        hist = [ep.state[max(t - k, 0)] for k in range(self.state_hist - 1, -1, -1)]
        return {"e": e, "t": t, "state": np.concatenate(hist).astype(np.float32),
                "action": ep.action[idx].astype(np.float32)}

    @staticmethod
    def collate(samples):
        return {"e": [s["e"] for s in samples], "t": [s["t"] for s in samples],
                "state": torch.from_numpy(np.stack([s["state"] for s in samples])),
                "action": torch.from_numpy(np.stack([s["action"] for s in samples]))}

    def featurize_batch(self, batch, featurizer, device="cuda"):
        """Run the perception on every (episode, frame) of the batch -> the policy's observation dict."""
        feats = []
        for e, t in zip(batch["e"], batch["t"]):
            ep = self.eps[e]
            if getattr(featurizer, "_ep", None) is not ep:          # stabilisers / table height are per episode
                featurizer.reset(ep); featurizer._ep = ep
            feats.append(featurizer.frame(ep, t))
        xyz = [np.asarray(f["xyz"], np.float32) for f in feats]; pts = [np.asarray(f["pcd_pts"], np.float32) for f in feats]
        if self.pcd_ee_relative:                                    # as CacheDataset: minus the TCP of the current state
            st = batch["state"].numpy()
            for i, f in enumerate(feats):
                xyz[i], pts[i] = self._relativize(xyz[i], pts[i], f["pcd_mask"], self._tcp(st[i, -self.state_dim:][:7].astype(np.float64)), ref=self.pcd_ref)
        stack = lambda k, dt=np.float32: torch.from_numpy(np.stack([f[k] for f in feats]).astype(dt)).to(device, non_blocking=True)
        obs = {"patch_tokens": stack("patches"), "box2d": stack("box2d"), "xyz": torch.from_numpy(np.stack(xyz)).to(device), "boxgeo": stack("boxgeo"),
               "pcd_pts": torch.from_numpy(np.stack(pts)).to(device), "pcd_mask": torch.from_numpy(np.stack([f["pcd_mask"] for f in feats])).to(device),
               "state": batch["state"].to(device), "lang": torch.zeros(len(feats), 768, device=device)}
        return obs, batch["action"].to(device)
