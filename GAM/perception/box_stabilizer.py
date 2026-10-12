"""Causal temporal stabilizer for per-slot WildDet detections (2D box + 3D box).

Shared by offline precompute and online deployment so training features and inference
features pass through IDENTICAL filtering (no train/eval distribution split). Strictly
causal: current frame + history only.

2D box: EMA on (cx, cy, w, h), gated by IoU with the current state, a detection that
    does not overlap is an outlier (state held); after `miss_reset` consecutive misses it
    is accepted as a real jump and the state resets (fast object motion / target switch).
3D box: center EMA; extents EMA (rigid objects -> converges then near-frozen); rotation
    blended slowly and re-orthonormalized via SVD; >jump_m center jumps rejected.
Call reset(key) (or use a fresh key) at segment boundaries.
"""
from __future__ import annotations

import numpy as np


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return float(i / max(ua, 1e-9))


class BoxStabilizer:
    def __init__(self, a_c2d: float = 0.3, a_s2d: float = 0.15, iou_gate: float = 0.2,
                 miss_reset: int = 3, a_c3d: float = 0.3, a_e3d: float = 0.1,
                 a_r3d: float = 0.2, jump_m: float = 0.15):
        self.p = dict(a_c2d=a_c2d, a_s2d=a_s2d, iou_gate=iou_gate, miss_reset=miss_reset,
                      a_c3d=a_c3d, a_e3d=a_e3d, a_r3d=a_r3d, jump_m=jump_m)
        self.s = {}

    def reset(self, key=None):
        if key is None:
            self.s = {}
        else:
            self.s.pop(key, None)

    def __call__(self, key, det):
        """det: Detection (box2d, optionally center/extents/R). Smoothed in place and returned."""
        p = self.p
        st = self.s.get(key)
        has3d = det.center is not None
        if st is None:
            st = {"b": det.box2d.astype(np.float64).copy(), "miss": 0,
                  "c": det.center.copy() if has3d else None,
                  "e": det.extents.copy() if has3d else None,
                  "R": det.R.copy() if has3d else None}
            self.s[key] = st
        else:
            # ---- 2D ----
            if det.box2d[2] > det.box2d[0]:
                if _iou(st["b"], det.box2d) >= p["iou_gate"]:
                    st["miss"] = 0
                    nb = det.box2d.astype(np.float64)
                    oc = [(st["b"][0] + st["b"][2]) / 2, (st["b"][1] + st["b"][3]) / 2]
                    ow, oh = st["b"][2] - st["b"][0], st["b"][3] - st["b"][1]
                    nc = [(nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2]
                    nw, nh = nb[2] - nb[0], nb[3] - nb[1]
                    c = [oc[0] + p["a_c2d"] * (nc[0] - oc[0]), oc[1] + p["a_c2d"] * (nc[1] - oc[1])]
                    w = ow + p["a_s2d"] * (nw - ow); h = oh + p["a_s2d"] * (nh - oh)
                    st["b"] = np.array([c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2])
                else:
                    st["miss"] += 1
                    if st["miss"] >= p["miss_reset"]:   # a real jump: accept and reset
                        st["b"] = det.box2d.astype(np.float64).copy(); st["miss"] = 0
                        if has3d:
                            st["c"], st["e"], st["R"] = det.center.copy(), det.extents.copy(), det.R.copy()
            # ---- 3D ----
            if has3d and st["c"] is not None:
                if np.linalg.norm(det.center - st["c"]) < p["jump_m"]:
                    st["c"] = (1 - p["a_c3d"]) * st["c"] + p["a_c3d"] * det.center
                    st["e"] = (1 - p["a_e3d"]) * st["e"] + p["a_e3d"] * det.extents
                    M = (1 - p["a_r3d"]) * st["R"] + p["a_r3d"] * det.R
                    U, _, Vt = np.linalg.svd(M); st["R"] = U @ Vt
            elif has3d:
                st["c"], st["e"], st["R"] = det.center.copy(), det.extents.copy(), det.R.copy()
        det.box2d = st["b"].astype(np.float32)
        if st["c"] is not None:
            det.center = st["c"].astype(np.float32)
            det.extents = st["e"].astype(np.float32)
            det.R = st["R"].astype(np.float32)
        return det
