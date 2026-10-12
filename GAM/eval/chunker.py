"""One action per control step from a chunk policy, the three ways of stitching successive chunks.

    chunker = ActionChunker.from_config(cfg["eval"], gripper_dims=(6, 13))
    chunker.reset()                                   # at the start of every episode
    while not done:
        a = chunker.act(lambda prefix: runner.act(obs, prefix)[0])   # predict(prefix) -> chunk (H, Da)
        env.step(a)

replan   execute the first `exec_steps` actions of each chunk, then re-plan; the first action of the new chunk is
         averaged with what the previous chunk predicted for that step (arm joints only: `seam_blend`)
tempens  temporal ensembling: re-plan every `query_every` steps, execute the weighted mean of every prediction
         made for the current step (weights exp(-k * age), newest heaviest); gripper dims take the newest
rtc      real-time chunking: like replan, but the unexecuted tail of the previous chunk is handed to the policy as
         a soft prefix (`predict(prefix)`), with its first `delay` actions frozen (inference latency, in steps)

`predict` is called only when a new chunk is needed (`needs_query()` says when), so a harness that re-renders
its observation per step can skip the work in between.
"""
from __future__ import annotations

import numpy as np


class ActionChunker:
    def __init__(self, mode: str = "replan", exec_steps: int = 15, seam_blend: bool = True, k: float = 0.01,
                 query_every: int = 1, delay: int = 1, gripper_dims=()):
        assert mode in ("replan", "tempens", "rtc"), mode
        self.mode, self.exec_steps, self.seam_blend = mode, int(exec_steps), bool(seam_blend)
        self.k, self.query_every, self.delay = float(k), max(1, int(query_every)), int(delay)
        self.gripper_dims = tuple(int(i) for i in gripper_dims)
        self.reset()

    @classmethod
    def from_config(cls, ev: dict, gripper_dims=()):
        """eval section of a config: chunking + the replan / tempens / rtc sub-sections."""
        mode = ev.get("chunking", "replan"); sub = ev.get(mode) or {}
        return cls(mode, exec_steps=int(sub.get("exec_steps", 15 if mode == "replan" else 8)),
                   seam_blend=bool(sub.get("seam_blend", True)), k=float(sub.get("k", 0.01)),
                   query_every=int(sub.get("query_every", 1)), delay=int(sub.get("delay", 1)),
                   gripper_dims=tuple(ev.get("gripper_dims", gripper_dims) or ()))   # eval.gripper_dims: [6, 13] bimanual, [7] single-arm Franka

    def reset(self):
        self.queue, self.tail, self.seam, self.buf, self.t = [], None, None, {}, 0

    def needs_query(self) -> bool:
        return (self.t % self.query_every == 0) if self.mode == "tempens" else not self.queue

    def act(self, predict) -> np.ndarray:
        if self.mode == "tempens":
            if self.t % self.query_every == 0:
                for i, a in enumerate(np.asarray(predict(None), np.float32)):
                    self.buf.setdefault(self.t + i, []).append(a)
            preds = self.buf.pop(self.t, None)
            if not preds:                                          # chunk shorter than query_every: re-plan now
                for i, a in enumerate(np.asarray(predict(None), np.float32)):
                    self.buf.setdefault(self.t + i, []).append(a)
                preds = self.buf.pop(self.t)
            w = np.exp(-self.k * np.arange(len(preds)))[::-1]; w = w / w.sum()   # newest prediction weighs most
            a = (np.stack(preds) * w[:, None]).sum(0)
            grip = [i for i in self.gripper_dims if i < len(a)]; a[grip] = preds[-1][grip]
            self.t += 1
            return a
        if not self.queue:
            chunk = np.asarray(predict(self.tail if self.mode == "rtc" else None), np.float32).copy()
            n = max(min(self.exec_steps, len(chunk) - 1), 1)
            if self.mode == "replan" and self.seam_blend and self.seam is not None and len(chunk) > 1:
                grip = [i for i in self.gripper_dims if i < chunk.shape[1]]; g = chunk[0][grip].copy()
                chunk[0] = 0.5 * (self.seam + chunk[0]); chunk[0][grip] = g
            if len(chunk) > n:
                self.seam = chunk[n].copy(); self.tail = chunk[n:].copy()
            self.queue = list(chunk[:n])
        self.t += 1
        return self.queue.pop(0)


def predict_with_prefix(policy, obs, prefix, delay: int = 1, generator=None):
    """GAMPolicy chunk for one observation; `prefix` (K, Da) = the previous chunk's unexecuted tail (rtc) or None."""
    import torch
    if prefix is not None and len(prefix):
        dev = obs["state"].device
        pf = torch.as_tensor(np.asarray(prefix, np.float32))[None].to(dev); K = pf.shape[1]
        w = policy.rtc_weights(policy.horizon, K, delay).to(dev)
        out = policy.predict_action(obs, generator=generator, prefix=pf, prefix_weights=w, n_frozen=min(delay, K))
    else:
        out = policy.predict_action(obs, generator=generator)
    return out[0].float().cpu().numpy()
