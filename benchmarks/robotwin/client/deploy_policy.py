"""RoboTwin policy client for GAM. Lives in <RoboTwin>/policy/gam (see ../eval.py, which links it).

Two ways to reach the policy, chosen by the environment variable GAM_SERVER_URL:
  set   -> HTTP client: observations are posted to ../server.py running in the GAM environment
           (use this when RoboTwin and GAM live in different Python environments)
  unset -> in-process: the policy is built here from GAM_CONFIG / GAM_CKPT (one environment for both)

Either way the head-camera RGB-D, the camera calibration and the joint state go in and an action
chunk comes out. How successive chunks are combined comes from the `eval` section of the task config:

  replan  : execute the first `exec_steps` actions of each chunk, then re-plan; the first action of the
            new chunk is averaged with what the previous chunk predicted for that step (arm joints only)
  tempens : re-plan every `query_every` steps and average overlapping predictions with weights exp(-k * age)
  rtc     : like replan, but the unexecuted tail of the previous chunk constrains the new one (real-time chunking)

With GAM_OVERLAY_DIR set (eval.py --overlay) every episode is also written as <dir>/episode<i>_<S|F>.mp4: the head
camera with the detector's 3D boxes on every step (the detector is run per frame for the video; the policy
itself is still queried once per chunk).
"""
import atexit, base64, io, json, os, urllib.request

import numpy as np

GRIPPER_DIMS = (6, 13)
SLOT_COLORS = [(66, 135, 245), (52, 199, 89), (255, 159, 10), (191, 90, 242), (255, 69, 58), (100, 210, 255)]   # RGB
_SGN = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], np.float32)
_EDGES = [(i, i ^ m) for i in range(8) for m in (1, 2, 4) if i < i ^ m]


def draw_boxes(img, boxes, K, E, slots):
    """Wireframes of the backend's 3D boxes ({slot, p, extents, R}) on the head-camera image (same maths as GAM.perception.draw)."""
    import cv2
    K = np.asarray(K, np.float64).reshape(3, 3); E = np.asarray(E, np.float64).reshape(3, 4)
    for b in boxes:
        R = np.asarray(b["R"], np.float32).reshape(3, 3)
        c = np.asarray(b["p"], np.float32) + (_SGN * np.asarray(b["extents"], np.float32) / 2) @ R.T
        pc = c @ E[:, :3].T + E[:, 3]
        if (pc[:, 2] > 0.01).sum() < 8:
            continue
        uv = (pc[:, :2] / pc[:, 2:3]) @ K[:2, :2].T + K[:2, 2]; p = np.round(uv).astype(int)
        col = SLOT_COLORS[slots.index(b["slot"]) % len(SLOT_COLORS)] if b["slot"] in slots else SLOT_COLORS[0]
        for i, j in _EDGES:
            cv2.line(img, tuple(p[i]), tuple(p[j]), col, 2, cv2.LINE_AA)
    return img


def write_video(path, frames, fps=10):
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, quality=7)
    except ImportError:
        import cv2
        H, W = frames[0].shape[:2]; vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for f in frames:
            vw.write(f[:, :, ::-1])
        vw.release()


def _b64(a):
    b = io.BytesIO(); np.savez_compressed(b, a=np.ascontiguousarray(a)); return base64.b64encode(b.getvalue()).decode()


def _request(observation, prefix):
    hc = observation["observation"]["head_camera"]
    return {"rgb": _b64(np.asarray(hc["rgb"])), "depth": _b64(np.asarray(hc["depth"], np.float32)),
            "K": np.asarray(hc["intrinsic_cv"], np.float64).reshape(-1).tolist(),
            "E": np.asarray(hc["extrinsic_cv"], np.float64).reshape(-1).tolist(),
            "state": [float(v) for v in np.asarray(observation["joint_action"]["vector"]).reshape(-1)],
            "prefix": (np.asarray(prefix, np.float32).tolist() if prefix is not None else None)}


class HttpBackend:
    def __init__(self, url):
        self.url = url

    def __call__(self, path, payload):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        if "error" in r:
            raise RuntimeError(r["error"])
        return r


class InProcessBackend:
    """The same Server object as server.py, called directly."""

    def __init__(self, config, ckpt=None):
        import sys
        here = os.path.dirname(os.path.realpath(__file__)); sys.path.insert(0, os.path.dirname(here))
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(here))))
        from GAM.config import load_config
        from GAM.hub import fetch
        from server import Server
        cfg = load_config(config)
        self.server = Server(cfg, fetch(ckpt or cfg["checkpoints"]["policy"]))

    def __call__(self, path, payload):
        return {"/reset": self.server.reset, "/boxes": self.server.boxes}.get(path, self.server.act)(payload)


class GAMClient:
    def __init__(self, usr_args):
        url = os.environ.get("GAM_SERVER_URL")
        self.backend = HttpBackend(url) if url else InProcessBackend(os.environ["GAM_CONFIG"], os.environ.get("GAM_CKPT"))
        self.n_ep = -1; self.ready = False; self.cfg = {}; self.slots = []
        self.overlay = os.environ.get("GAM_OVERLAY_DIR"); self.frames = []; self.boxes = []; self.succ = False
        if self.overlay:
            os.makedirs(self.overlay, exist_ok=True); atexit.register(self.flush)
        self._clear()

    def _clear(self):
        self.seam = None; self.tail = None; self.buf = {}; self.tt = 0

    def flush(self):
        """Write the overlay video of the episode just finished."""
        if self.overlay and self.frames:
            write_video(os.path.join(self.overlay, f"episode{self.n_ep}_{'S' if self.succ else 'F'}.mp4"), self.frames)
        self.frames, self.boxes, self.succ = [], [], False

    def reset(self, instruction):
        self.flush(); self.n_ep += 1
        r = self.backend("/reset", {"instruction": instruction, "seed": self.n_ep}); self.cfg = r["eval"]; self.slots = list(r.get("slots", []))
        self.ready = True; self._clear()

    def act(self, observation, prefix=None):
        r = self.backend("/act", _request(observation, prefix)); self.boxes = r.get("boxes", [])
        return np.asarray(r["actions"], np.float32)

    def record(self, observation, success=False):
        """One overlay frame: the head camera of this observation with the boxes the detector finds on it
        (detection only, the policy is not queried, so the video does not change the rollout)."""
        if self.overlay:
            hc = observation["observation"]["head_camera"]
            boxes = self.backend("/boxes", _request(observation, None)).get("boxes", self.boxes)
            self.frames.append(draw_boxes(np.ascontiguousarray(hc["rgb"]).copy(), boxes, hc["intrinsic_cv"], hc["extrinsic_cv"], self.slots))
            self.succ = self.succ or bool(success)


def encode_obs(observation):
    return observation


def get_model(usr_args):
    return GAMClient(usr_args)


def reset_model(model):
    model.ready = False; model._clear()


def eval(TASK_ENV, model, observation):  # noqa: A001
    if not model.ready:
        model.reset(TASK_ENV.get_instruction())
    ev = model.cfg; mode = ev.get("chunking", "replan")

    def step(a):
        TASK_ENV.take_action(a, action_type="qpos")
        if model.overlay:                                         # re-render the head camera for the overlay video
            model.record(TASK_ENV.get_obs(), TASK_ENV.eval_success)
    if mode == "tempens":
        te = ev.get("tempens") or {}; k = float(te.get("k", 0.01))
        chunk = model.act(observation)
        for i, a in enumerate(chunk):
            model.buf.setdefault(model.tt + i, []).append(a)
        for _ in range(int(te.get("query_every", 4))):
            preds = model.buf.pop(model.tt, None)
            if not preds:
                break
            w = np.exp(-k * np.arange(len(preds)))[::-1]; w = w / w.sum()       # newest prediction weighs most
            a = (np.stack(preds) * w[:, None]).sum(0)
            grip = [i for i in GRIPPER_DIMS if i < len(a)]; a[grip] = preds[-1][grip]
            step(a); model.tt += 1
        return
    rtc = mode == "rtc"
    n_exec = int(((ev.get("rtc") if rtc else ev.get("replan")) or {}).get("exec_steps", 15))
    chunk = model.act(observation, prefix=(model.tail if rtc else None)).copy()
    n_exec = max(min(n_exec, len(chunk) - 1), 1)
    if not rtc and (ev.get("replan") or {}).get("seam_blend", True) and model.seam is not None and len(chunk) > 1:
        grip = [i for i in GRIPPER_DIMS if i < chunk.shape[1]]; g = chunk[0][grip].copy()
        chunk[0] = 0.5 * (model.seam + chunk[0]); chunk[0][grip] = g
    if len(chunk) > n_exec:
        model.seam = chunk[n_exec].copy(); model.tail = chunk[n_exec:].copy()
    for a in chunk[:n_exec]:
        step(a)
