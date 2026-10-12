"""Environment adapters for GAM.eval.run.

Write one for your robot or simulator (any module, passed as --env my_robot.py:MyEnv):

    class MyEnv:
        n_episodes = 20                       # optional: how many episodes --env can offer
        def __init__(self, cfg): ...          # the yaml (data paths, robot URDF, camera calibration ...)
        def reset(self, episode) -> str: ...  # start episode `episode`; return its instruction ("" if none)
        def observe(self) -> dict: ...        # {"rgb": (H,W,3) uint8 RGB, "depth": (H,W) float m (or mm; zeros if none),
                                              #  "K": (3,3), "E": (3,4) base->camera (OpenCV), "state": (Dq,) joint vector}
        def step(self, action) -> (done, success): ...   # execute one action of the checkpoint's layout
        def close(self): ...

ReplayEnv (built in, --env replay) plays the recorded episodes of data.episodes_glob / the LeRobot dataset back,
frame by frame: the policy sees the recorded observations, its actions are compared with the recorded ones
(action_mae in results.jsonl) and are not executed, so `success` is always False. It is open loop, a smoke test of
a freshly trained checkpoint on its own data (does it track the objects, does it imitate the demonstrations) and
the way to get a 3D-box video without a simulator; closed-loop success needs a real environment.
"""
from __future__ import annotations

import numpy as np


class ReplayEnv:
    def __init__(self, cfg: dict):
        from GAM.data.episodes import list_episodes, load_episode
        self.cfg = dict(cfg, data=dict(cfg["data"], clicks={"source": "none"}))
        self.paths = list_episodes(self.cfg); self.n_episodes = len(self.paths)
        self._load = load_episode; self.ep = None; self.t = 0; self.T = 0
        self.swap = bool(cfg.get("perception", {}).get("image_channel_swap", False))   # recordings stored BGR -> hand out RGB like a live camera

    def reset(self, episode: int) -> str:
        if self.ep is not None:
            self.ep.close()
        self.ep = self._load(self.paths[episode], self.cfg); self.t = 0; self.T = self.ep.T
        return self.ep.instruction

    def observe(self) -> dict:
        t = min(self.t, self.T - 1); rgb = self.ep.rgb(t)
        return {"rgb": (rgb[:, :, ::-1] if self.swap else rgb), "depth": self.ep.depth_m(t), "K": self.ep.K[t], "E": self.ep.E[t],
                "state": self.ep.state[t], "action": self.ep.action[t]}

    def step(self, action):
        self.t += 1
        return self.t >= self.T, False

    def close(self):
        if self.ep is not None:
            self.ep.close()
