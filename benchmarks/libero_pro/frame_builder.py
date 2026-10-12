"""Per-frame observation builder for LIBERO, the single source of policy inputs, used by both
the training-cache precompute and the closed-loop evaluation.

Inputs per frame: agentview RGB + depth, camera calibration, joint angles, the instruction.
No simulator ground truth (object poses, segmentation, goal regions) is read.

  instruction --text_slots--> up to four text prompts [pick0, place0, pick1, place1]
  grounding "text+track": frame 0  WildDet3D text detection -> box centres -> CoTracker follows them
            "text":       WildDet3D text detection on every frame
  per frame: WildDet3D (prompted by the tracked points / texts) -> patch tokens, 2D + 3D boxes, depth
             robot mask from the robot model + joint angles + depth   (arm_mask.ProjectedRobotMask)
             object geometry + 512-point clouds in the robot base frame

build() returns  patches (G*G,1024) / box2d (4,4) / xyz (4,3) / boxgeo (4,9) / pcd_pts (4,N,3) /
                 pcd_mask (4,N) / clicks (4,2) / rgb.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_mask import ProjectedRobotMask
from sim_utils import cam_K, cam_ext_cv, base_T, real_depth
from text_slots import TextSlots

SLOTS = ["pick0", "place0", "pick1", "place1"]


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    it = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it
    return it / ua if ua > 0 else 0.0


class FrameBuilder:
    """fb = FrameBuilder(sim, cfg["perception"], wilddet_ckpt, cotracker_ckpt)
    fb.set_task(instruction); fb.reset(sim) per episode; fb.build(sim, obs) per frame."""

    def __init__(self, sim, pc: dict, wilddet_ckpt: str, cotracker_ckpt: str = None, device: str = "cuda"):
        self.res = int(pc.get("image_size", 512)); self.grid = int(pc.get("patch_grid", 6))
        self.n_points = int(pc.get("n_points", 512))
        self.grounding = pc.get("grounding", "text+track"); assert self.grounding in ("text+track", "text"), self.grounding
        self.detector_depth = pc.get("detector_depth", "sim"); assert self.detector_depth in ("sim", "zero")
        self.geometry_depth = pc.get("geometry_depth", "sim"); assert self.geometry_depth in ("sim", "pred")
        self.geometry_source = pc.get("geometry_source", "box2d_depth")
        self.deterministic = bool(pc.get("deterministic", True))
        self.tracker_ckpt, self.device = cotracker_ckpt, device
        self.stagger = int(pc.get("tracker_stagger", 1))
        self.text_slots = TextSlots(pc.get("text_slots", "table"), pc.get("span_cache"), pc.get("lang_slots"),
                                    span_head=pc.get("span_head"), device=device)
        self.prompts = None; self.click_shift = None
        self.offline_tracks = None                     # (T,K,2) px + (T,K) visibility from track_offline()

        from GAM.config import build_extractor
        self.ext = build_extractor(dict(pc, wilddet3d=wilddet_ckpt), grid=self.grid, device=device)
        # the depth branch was trained with random depth dropout; make inference deterministic
        gb = self.ext.pred.wilddet3d.geometry_backend
        gb.monocular_prob = 1.0 if self.geometry_depth == "pred" else 0.0
        gb.masked_prob = 0.0
        self.K = cam_K(sim, "agentview", self.res, self.res)
        self.reset(sim)

    # ---- task / episode state ------------------------------------------------------------
    def set_task(self, instruction: str, click_shift=None):
        """click_shift: None or (dx, dy, slots), offset of the tracker's start point from the box centre."""
        self.prompts, spans = self.text_slots(instruction)
        self.click_shift = click_shift
        print(f"[lang] {instruction!r} spans={spans} -> slots={self.prompts}", flush=True)

    def reset(self, sim):
        """New episode (env.reset rebuilds the sim): re-derive the base pose and the robot model."""
        self.baseT, self.baseTi, _ = base_T(sim)
        self.arm = ProjectedRobotMask(sim)
        self.table_z = None
        self.tracker = None; self.tk_slots = []; self.frame_i = 0; self._hold = {}

    # ---- helpers -------------------------------------------------------------------------
    def _detector_depth(self, dep):
        if self.geometry_depth != "pred" or self.detector_depth == "sim":
            return dep
        return np.zeros_like(dep)

    def _text_boxes(self, rgb, dep, K, texts):
        """Text-prompted detection of each phrase on one frame -> per phrase, boxes (xyxy px) by descending score."""
        import torch, cv2
        from GAM.perception.patch_tokens import INPUT, IMEAN, ISTD
        H0, W0 = rgb.shape[:2]; sx, sy = INPUT / W0, INPUT / H0; N = len(texts)
        im = cv2.resize(rgb, (INPUT, INPUT))
        if getattr(self.ext, "channel_swap", False):
            im = im[:, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float()[None] / 255.0
        x = ((x - IMEAN) / ISTD).to(self.ext.dev)
        Ks = torch.from_numpy((np.diag([sx, sy, 1.0]) @ K).astype(np.float32)).to(self.ext.dev)
        dgt = torch.from_numpy(cv2.resize(np.asarray(dep, np.float32), (INPUT, INPUT), interpolation=cv2.INTER_NEAREST)).to(self.ext.dev)[None, None]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=bool(getattr(self.ext, "bf16", True))):
            out = self.ext.pred(images=x.expand(N, -1, -1, -1), intrinsics=Ks[None].expand(N, -1, -1),
                                input_hw=[(INPUT, INPUT)] * N, original_hw=[(INPUT, INPUT)] * N, padding=[(0, 0, 0, 0)] * N,
                                input_texts=list(texts), depth_gt=dgt.expand(N, -1, -1, -1))
        res = []
        for i in range(N):
            b2, sc = out[0][i], out[2][i]
            if not (torch.is_tensor(sc) and sc.numel()):
                res.append([]); continue
            lst = []
            for j in torch.argsort(sc, descending=True).tolist()[:8]:
                b = b2[j].float().cpu().numpy() / INPUT * [W0, H0, W0, H0]
                if b[2] > b[0] and b[3] > b[1]:
                    lst.append(b.astype(np.float32))
            res.append(lst)
        return res

    def track_offline(self, frames_rgb, clicks, ckpt):
        """Whole-episode CoTracker (offline model) from the frame-0 clicks; build() then follows these
        tracks instead of the online tracker. Used when precomputing the training cache."""
        import torch
        from cotracker.predictor import CoTrackerPredictor
        if not hasattr(self, "_offline_p"):
            self._offline_p = CoTrackerPredictor(checkpoint=ckpt, offline=True, window_len=60).to(self.device).eval()
        vid = torch.from_numpy(np.stack(frames_rgb)).permute(0, 3, 1, 2).float()[None].to(self.device)
        q = torch.tensor([[[0.0, float(x), float(y)] for x, y in clicks]], device=self.device)
        with torch.no_grad():
            tr, vi = self._offline_p(vid, queries=q)
        self.offline_tracks = (tr[0].float().cpu().numpy(), vi[0].float().cpu().numpy() > 0.5)
        return self.offline_tracks

    def _first_frame_points(self, rgb, dep, K):
        """Frame 0 of "text+track": detect every prompt, pick one box per slot, return its (shifted) centre."""
        uniq = []
        for t in self.prompts:
            if t and t not in uniq:
                uniq.append(t)
        blists = dict(zip(uniq, self._text_boxes(rgb, self._detector_depth(dep), K, uniq))) if uniq else {}
        taken = {t: [] for t in uniq}
        self.lang_boxes = {}; cl, cl_slot = [], []
        for k, t in enumerate(self.prompts):
            if not t:
                continue
            # "open the drawer ... and close it": a second mention with no place target is the same entity
            same_entity = (k >= 2 and t in self.prompts[:k] and not (self.prompts[k + 1] if k + 1 < 4 else None))
            if same_entity and taken[t]:
                b = taken[t][0]
            else:      # n-th mention of a phrase ("both X") -> n-th non-overlapping box; otherwise reuse the best one
                b = next((bb for bb in blists.get(t, []) if all(_iou(bb, o) < 0.5 for o in taken[t])), None)
                if b is None and blists.get(t):
                    b = blists[t][0]
            if b is None:
                print(f"[lang] frame0: no box for slot{k} {t!r}", flush=True); continue
            taken[t].append(b); self.lang_boxes[k] = b
            u, v = float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)
            if self.click_shift is not None and k in self.click_shift[2]:
                u = float(np.clip(u + self.click_shift[0], b[0] + 1, b[2] - 1)); v = float(np.clip(v + self.click_shift[1], b[1] + 1, b[3] - 1))
            cl.append((u, v)); cl_slot.append(k)
        return cl, cl_slot

    # ---- per frame -----------------------------------------------------------------------
    def _clean_depth(self, d, K, E, am, refresh=False):
        """Depth for the object geometry: robot pixels and the table (estimated once per episode) removed."""
        from GAM.perception.geometry import depth_to_world_grid
        g = d.copy(); g[am] = 0.0
        if self.table_z is None or refresh:
            P0, ok0 = depth_to_world_grid(g, K, E)
            hz, ez = np.histogram(P0[ok0][:, 2], bins=200, range=(-0.5, 1.0))
            self.table_z = float(ez[hz.argmax()] + (ez[1] - ez[0]) / 2)
        Pg, okg = depth_to_world_grid(g, K, E)
        g[okg & (Pg[..., 2] < self.table_z + 0.008)] = 0.0
        return g

    def _sense(self, sim, obs):
        """RGB, depth, calibration, robot mask and the cleaned geometry depth of one frame."""
        from GAM.perception.geometry import depth_to_world_grid
        RES, K = self.res, self.K
        _, _, E_w = cam_ext_cv(sim, "agentview")                    # world->cam 3x4
        E = E_w @ self.baseT                                         # base->cam
        self.last_E = E                                              # for drawing the boxes on the video
        rgb = np.flipud(obs["agentview_image"]).copy()
        dep = real_depth(sim, np.flipud(obs["agentview_depth"].squeeze(-1)).astype(np.float64)).astype(np.float32)
        am = self.arm(obs, dep, K, E_w)

        dep_geo = self._clean_depth(dep, K, E, am)
        return rgb, dep, K, E, am, dep_geo

    def _ground(self, sim, rgb, dep, K):
        """Object points for this frame: text detection (every frame or frame 0) or the tracker."""
        RES = self.res
        clicks = np.full((4, 2), np.nan, np.float32)
        cl, cl_slot, texts = [], [], None
        if self.grounding == "text":
            cl_slot = [k for k, t in enumerate(self.prompts) if t]; texts = [self.prompts[k] for k in cl_slot]
        elif self.frame_i == 0:
            cl, cl_slot = self._first_frame_points(rgb, dep, K)
            for (u, v), k in zip(cl, cl_slot):
                clicks[k] = (u, v)
            self.tk_slots = list(cl_slot)
            if self.offline_tracks is None:
                from GAM.perception.tracker import PointTracker
                self.tracker = PointTracker(self.tracker_ckpt, self.device, track_wh=(RES, RES), stagger=self.stagger)
                assert self.tracker.available, "CoTracker unavailable"
                self.tracker.start(rgb, cl)
        else:
            if self.offline_tracks is not None:
                tr, vi = self.offline_tracks
                fi = min(self.frame_i, len(tr) - 1)
                pts = [tuple(map(float, tr[fi, i])) for i in range(len(self.tk_slots))]; vis = [bool(v) for v in vi[fi]]
            else:
                pts = self.tracker.push(rgb) if self.tracker is not None else []; vis = self.tracker.vis if self.tracker is not None else []
            for i, k in enumerate(self.tk_slots):
                if i < len(pts) and vis[i]:
                    u, v = pts[i]
                    if 0 <= u < RES and 0 <= v < RES:
                        cl.append((float(u), float(v))); cl_slot.append(k); clicks[k] = (u, v)
        return cl, cl_slot, texts, clicks

    def build(self, sim, obs, seed=0):
        """One closed-loop frame: sense -> ground (tracker state advances) -> features."""
        rgb, dep, K, E, am, dep_geo = self._sense(sim, obs)
        cl, cl_slot, texts, clicks = self._ground(sim, rgb, dep, K)
        self.frame_i += 1
        return self._features(rgb, dep, K, E, am, dep_geo, cl, cl_slot, texts, clicks, seed)

    def build_from_points(self, sim, obs, points, seed=0):
        """One frame with the object points given (e.g. stored offline tracks, (4,2) px, NaN = not visible);
        no tracker state, so frames can be visited in any order (end-to-end training)."""
        rgb, dep, K, E, am, dep_geo = self._sense(sim, obs)
        clicks = np.asarray(points, np.float32).reshape(4, 2); cl, cl_slot = [], []
        for k in range(4):
            u, v = clicks[k]
            if np.isfinite(u) and np.isfinite(v) and 0 <= u < self.res and 0 <= v < self.res:
                cl.append((float(u), float(v))); cl_slot.append(k)
        self._hold = {}                                   # no carry-over between unrelated frames
        return self._features(rgb, dep, K, E, am, dep_geo, cl, cl_slot, None, clicks, seed)

    def ground_first_frame(self, sim, obs):
        """Frame-0 text detection -> object points (4,2) px (NaN where no box); used by track.py."""
        rgb, dep, K, E, am, dep_geo = self._sense(sim, obs)
        cl, cl_slot = self._first_frame_points(rgb, dep, K)
        pts = np.full((4, 2), np.nan, np.float32)
        for (u, v), k in zip(cl, cl_slot):
            pts[k] = (u, v)
        return pts, rgb

    def _features(self, rgb, dep, K, E, am, dep_geo, cl, cl_slot, texts, clicks, seed):
        from GAM.perception.geometry import build_slot_obs, depth_to_world, depth_to_world_grid, object_from_box
        RES, K = self.res, K
        r = self.ext(rgb, self._detector_depth(dep), K, E, cl, arm_mask=am, texts=texts)
        if texts:
            for j, k in enumerate(cl_slot):
                b = r.detections[j].box2d
                if b[2] > b[0]:
                    clicks[k] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        if self.geometry_depth == "pred":                            # geometry from the detector's own depth
            dep_geo = self._clean_depth(np.asarray(self.ext.last_depth, np.float32), K, E, am, refresh=self.frame_i <= 1)
        out = {"patches": r.patches.numpy().astype(np.float16), "clicks": clicks,
               "box2d": np.zeros((4, 4), np.float32), "xyz": np.zeros((4, 3), np.float32),
               "boxgeo": np.zeros((4, 9), np.float32),
               "pcd_pts": np.zeros((4, self.n_points, 3), np.float16), "pcd_mask": np.zeros((4, self.n_points), bool)}
        objects, det_box = {}, {}
        for j, k in enumerate(cl_slot):
            d = r.detections[j]
            if self.geometry_source == "box3d" and d.center is not None and d.box2d[2] > d.box2d[0]:
                objects[SLOTS[k]] = {"p": np.asarray(d.center, np.float32), "extents": np.asarray(d.extents, np.float32),
                                     "R": np.asarray(d.R, np.float32)}
                det_box[k] = d.box2d.astype(np.float32).copy()
            elif d.box2d[2] > d.box2d[0]:
                obj = object_from_box(d.box2d, dep_geo, K, E, z_band=(0.06 if SLOTS[k].startswith("pick") else 0.15))
                if obj is not None:
                    objects[SLOTS[k]] = obj; det_box[k] = d.box2d.astype(np.float32).copy()
        if objects:
            so = build_slot_obs(depth_to_world(dep_geo, K, E), objects, SLOTS, self.n_points, E=E, K=K, seed=int(seed))
            for k in range(4):
                if SLOTS[k] in objects:
                    out["box2d"][k] = det_box[k]
                    out["xyz"][k] = so["xyz"][k]; out["boxgeo"][k] = so["boxgeo"][k]
                    out["pcd_pts"][k] = so["pcd_pts"][k]; out["pcd_mask"][k] = so["pcd_mask"][k]
        for k in range(4):                 # a slot that is empty this frame keeps its last valid observation
            if out["pcd_mask"][k].any():
                self._hold[k] = {key: out[key][k].copy() for key in ("xyz", "boxgeo", "box2d", "pcd_pts", "pcd_mask", "clicks")}
            elif k in self._hold:
                for key, v in self._hold[k].items():
                    out[key][k] = v
        out["rgb"] = rgb
        return out
