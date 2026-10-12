"""End-to-end training data for LIBERO: every sample is one demonstration frame, replayed in the
simulator and featurised on the spot (frozen WildDet3D + robot mask + object geometry), the same
code as evaluation. Nothing is precomputed except the object tracks (benchmarks/libero_pro/track.py:
frame-0 grounding + offline CoTracker, a few KB per demonstration), because tracking needs the whole
video. Frames are sampled at random like a normal dataset; one simulator instance is kept per task.

    python benchmarks/libero_pro/track.py --config configs/libero_pro.yaml --suite object    # once per suite
    python -m GAM.train --config configs/libero_pro.yaml --e2e
"""
from __future__ import annotations

import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))


class LiberoE2EDataset(Dataset):
    def __init__(self, cfg: dict, horizon: int, state_hist: int, device: str = "cuda"):
        import sys
        sys.path.insert(0, HERE)
        from frame_builder import FrameBuilder
        from GAM.hub import fetch
        self.cfg, self.horizon, self.state_hist, self.device = cfg, horizon, state_hist, device
        pc, ck, dt, tr = cfg["perception"], cfg["checkpoints"], cfg["data"], cfg["train"]
        self.res = int(pc.get("image_size", 512))
        self.wd_ckpt, self.tracker_ckpt = fetch(ck["wilddet3d"]), fetch(ck["cotracker"])
        self.FrameBuilder = FrameBuilder
        tracks_dir = tr["tracks_dir"]
        assert os.path.isdir(tracks_dir), f"{tracks_dir} not found: run benchmarks/libero_pro/track.py first (train.tracks_dir)"
        self.episodes = []
        for tdir in sorted(glob.glob(os.path.join(tracks_dir, "libero_*__*"))):
            suite, name = os.path.basename(tdir).split("__", 1)
            hdf5 = os.path.join(dt["demos_root"], suite, f"{name}_demo.hdf5")
            bddl = glob.glob(os.path.join(fetch(cfg["paths"]["task_files"].rstrip("/") + "/"), "bddl_files", suite, name + ".bddl"))
            assert bddl and os.path.exists(hdf5), f"missing demos/bddl for {suite}/{name}"
            with h5py.File(hdf5, "r") as f:
                for tp in sorted(glob.glob(os.path.join(tdir, "demo_*.npz")), key=lambda p: int(os.path.basename(p)[5:-4])):
                    z = np.load(tp); dk = os.path.basename(tp)[:-4]; idx = z["kept"]
                    g = f[f"data/{dk}"]; actions = g["actions"][:].astype(np.float32)[idx]
                    joints = g["obs/joint_states"][:].astype(np.float32)[idx]; grip = g["obs/gripper_states"][:].astype(np.float32)[idx]
                    state8 = np.concatenate([joints, np.clip(grip[:, :1] / 0.04, 0.0, 1.0)], 1).astype(np.float32)
                    pts = z["points"].astype(np.float32)
                    # a slot that is not visible at t keeps its last visible point (what the online tracker does at test time)
                    for k in range(4):
                        last = None
                        for t in range(len(pts)):
                            if np.isfinite(pts[t, k]).all():
                                last = pts[t, k]
                            elif last is not None:
                                pts[t, k] = last
                    self.episodes.append(dict(bddl=bddl[0], instruction=str(z["instruction"]), hdf5=hdf5, demo=dk, idx=idx,
                                              points=pts, state=state8, action=actions))
        assert self.episodes, f"no tracks under {tracks_dir}"
        self.index = [(e, t) for e, ep in enumerate(self.episodes) for t in range(len(ep["idx"]))]
        self.action_dim, self.state_dim, self.image_wh = 7, 8, (float(self.res), float(self.res))
        self.pcd_ref = str(tr.get("pcd_ref", "none")); self.action_mode, self.pcd_ee_relative = "abs", self.pcd_ref != "none"
        if self.pcd_ref != "none":
            from GAM.data.cache_dataset import relativize, ee_fk
            self._relativize, self._tcp = relativize, ee_fk()
        self.envs, self.states, self.fb, self.fb_task = {}, {}, None, None
        self.max_envs = int(dt.get("e2e_max_envs", 40))
        import atexit; atexit.register(self.close)                 # close the simulators before the EGL contexts are torn down

    def close(self):
        for env in self.envs.values():
            try:
                env.close()
            except Exception:
                pass
        self.envs = {}

    def __len__(self):
        return len(self.index)

    def all_actions(self):
        return np.concatenate([ep["action"] for ep in self.episodes], 0)

    def __getitem__(self, i):
        e, t = self.index[i]; ep = self.episodes[e]; T = len(ep["idx"])
        win = np.clip(np.arange(t, t + self.horizon), 0, T - 1)
        hist = [ep["state"][max(t - k, 0)] for k in range(self.state_hist - 1, -1, -1)]
        return {"e": e, "t": t, "state": np.concatenate(hist).astype(np.float32), "action": ep["action"][win].astype(np.float32)}

    @staticmethod
    def collate(samples):
        return {"e": [s["e"] for s in samples], "t": [s["t"] for s in samples],
                "state": torch.from_numpy(np.stack([s["state"] for s in samples])),
                "action": torch.from_numpy(np.stack([s["action"] for s in samples]))}

    # ---- simulator + perception --------------------------------------------------------
    def _env(self, bddl):
        if bddl not in self.envs:
            from libero.libero.envs import OffScreenRenderEnv
            if len(self.envs) >= self.max_envs:                  # drop the oldest simulator
                old = next(iter(self.envs)); self.envs.pop(old).close()
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=self.res, camera_widths=self.res, camera_depths=True)
            env.seed(0); env.reset(); self.envs[bddl] = env
        return self.envs[bddl]

    def _demo_states(self, ep):
        key = (ep["hdf5"], ep["demo"])
        if key not in self.states:
            if len(self.states) > 64:
                self.states.pop(next(iter(self.states)))
            with h5py.File(ep["hdf5"], "r") as f:
                self.states[key] = f[f"data/{ep['demo']}/states"][:]
        return self.states[key]

    def frame(self, e, t):
        ep = self.episodes[e]; env = self._env(ep["bddl"]); sim = env.env.sim
        obs = env.set_init_state(self._demo_states(ep)[ep["idx"][t]])
        if self.fb is None:
            self.fb = self.FrameBuilder(sim, dict(self.cfg["perception"], span_head=self.cfg["checkpoints"].get("span_head")), self.wd_ckpt, self.tracker_ckpt, device=self.device)
        if self.fb_task != ep["bddl"]:                          # base pose / robot model follow the simulator instance
            self.fb.set_task(ep["instruction"]); self.fb.reset(sim); self.fb_task = ep["bddl"]
        return self.fb.build_from_points(sim, obs, ep["points"][t], seed=int(ep["idx"][t]))

    def featurize_batch(self, batch, featurizer=None, device="cuda"):
        """-> (obs, actions) for the policy; `featurizer` is unused (the FrameBuilder lives here)."""
        feats = [self.frame(e, t) for e, t in zip(batch["e"], batch["t"])]
        xyz = [f["xyz"] for f in feats]; pts = [f["pcd_pts"].astype(np.float32) for f in feats]
        if self.pcd_ref != "none":                              # object geometry relative to the end effector (as in CacheDataset)
            st = batch["state"].numpy()
            for i, f in enumerate(feats):
                xyz[i], pts[i] = self._relativize(xyz[i], pts[i], f["pcd_mask"], self._tcp(st[i, -8:-1].astype(np.float64)), ref=self.pcd_ref)
        stack = lambda k, dt=np.float32: torch.from_numpy(np.stack([f[k] for f in feats]).astype(dt)).to(device, non_blocking=True)
        obs = {"patch_tokens": stack("patches"), "box2d": stack("box2d"), "xyz": torch.from_numpy(np.stack(xyz).astype(np.float32)).to(device),
               "boxgeo": stack("boxgeo"), "pcd_pts": torch.from_numpy(np.stack(pts)).to(device),
               "pcd_mask": torch.from_numpy(np.stack([f["pcd_mask"] for f in feats])).to(device),
               "state": batch["state"].to(device), "lang": torch.zeros(len(feats), 768, device=device)}
        return obs, batch["action"].to(device)
