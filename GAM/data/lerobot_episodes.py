"""LeRobot datasets as GAM episodes (data.source: lerobot), reads the on-disk format directly
(parquet + mp4 via pyarrow and ffmpeg, codebase_version v2.1 and v3.0), so the `lerobot` package is not needed.

    data:
      source: lerobot
      repo_id: user/dataset                 # Hub dataset id (downloaded on first use) ...
      root: null                            # ... or a local dataset folder
      camera: observation.images.head       # RGB video feature
      depth_key: null                       # optional depth feature stored as an array (H,W); null = monocular
      depth_unit: m
      state_key: observation.state
      action_key: action
      camera_intrinsics: [[fx,0,cx],[0,fy,cy],[0,0,1]]    # at the stored image resolution
      camera_extrinsics: [[...3x4 base->camera, OpenCV...]] # or a 4x4 camera pose with extrinsic_is_cam2base: true

LeRobot datasets rarely store depth or calibration: the monocular recipe (detector_depth: zero,
geometry_depth: pred) needs neither; the calibration above is constant for a fixed camera.
The episode's task string is the instruction.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from GAM.data.episodes import Episode

_META = {}


class _Meta:
    """Episode table of one dataset: for every episode the data file, row range, video file and frame offset."""

    def __init__(self, root: str):
        import pyarrow.parquet as pq
        self.root = root
        info = json.load(open(os.path.join(root, "meta", "info.json")))
        self.fps = float(info["fps"]); self.features = info["features"]
        self.data_path, self.video_path = info["data_path"], info.get("video_path")
        self.version = str(info.get("codebase_version", "v2.1"))
        self.episodes = []
        if self.version.startswith("v3"):
            tasks = pq.read_table(os.path.join(root, "meta", "tasks.parquet")).to_pandas()
            self.task_of = {int(r.task_index): str(i) for i, r in tasks.iterrows()}
            for f in sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet"))):
                for r in pq.read_table(f).to_pylist():
                    vids = {k: dict(chunk=r[f"videos/{k}/chunk_index"], file=r[f"videos/{k}/file_index"],
                                    offset=int(round(float(r[f"videos/{k}/from_timestamp"]) * self.fps)))
                            for k, v in self.features.items() if v["dtype"] == "video"}
                    self.episodes.append(dict(index=int(r["episode_index"]), length=int(r["length"]),
                                              task=(r["tasks"] or [""])[0], data=dict(chunk=r["data/chunk_index"], file=r["data/file_index"]),
                                              lo=int(r["dataset_from_index"]), hi=int(r["dataset_to_index"]), videos=vids))
        else:                                               # v2.x: one parquet / mp4 per episode
            self.task_of = {int(j["task_index"]): j["task"] for j in map(json.loads, open(os.path.join(root, "meta", "tasks.jsonl")))}
            chunks = int(info.get("chunks_size", 1000))
            for j in map(json.loads, open(os.path.join(root, "meta", "episodes.jsonl"))):
                e = int(j["episode_index"])
                self.episodes.append(dict(index=e, length=int(j["length"]), task=(j.get("tasks") or [""])[0],
                                          data=dict(episode_chunk=e // chunks, episode_index=e), lo=None, hi=None,
                                          videos={k: dict(episode_chunk=e // chunks, episode_index=e, offset=0)
                                                  for k, v in self.features.items() if v["dtype"] == "video"}))
        self.episodes.sort(key=lambda d: d["index"])
        self._tables = {}

    def rows(self, ep):
        """pyarrow table of the episode's frames (in order)."""
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        if self.version.startswith("v3"):
            path = os.path.join(self.root, self.data_path.format(chunk_index=ep["data"]["chunk"], file_index=ep["data"]["file"]))
        else:
            path = os.path.join(self.root, self.data_path.format(**ep["data"]))
        if path not in self._tables:
            if len(self._tables) > 4:
                self._tables.pop(next(iter(self._tables)))
            self._tables[path] = pq.read_table(path)
        t = self._tables[path]
        t = t.filter(pc.equal(t["episode_index"], ep["index"]))
        return t.sort_by("frame_index")

    def video(self, ep, key):
        v = ep["videos"][key]
        if self.version.startswith("v3"):
            return os.path.join(self.root, self.video_path.format(video_key=key, chunk_index=v["chunk"], file_index=v["file"])), v["offset"]
        return os.path.join(self.root, self.video_path.format(video_key=key, **{k: v[k] for k in ("episode_chunk", "episode_index")})), 0


def _meta(cfg) -> _Meta:
    d = cfg["data"]; root = d.get("root")
    if not root:
        from GAM.hub import fetch
        root = fetch(f"hf://datasets/{d['repo_id']}/")
    if root not in _META:
        _META[root] = _Meta(root)
    return _META[root]


def list_episodes(cfg) -> list:
    m = _meta(cfg)
    return [f"lerobot://{m.root}#{ep['index']}" for ep in m.episodes]


class _Video:
    """Frames of one mp4 decoded with ffmpeg (imageio-ffmpeg's binary: handles the AV1 videos LeRobot writes,
    which OpenCV / decord builds often cannot). An episode's frames are decoded once and kept in memory."""

    def __init__(self, path, fps):
        self.path, self.fps = path, float(fps); self.cache = {}

    def frames(self, offset, n, H, W):
        key = (offset, n)
        if key not in self.cache:
            import subprocess
            import imageio_ffmpeg
            cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", self.path, "-ss", f"{offset / self.fps:.6f}",
                   "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]   # -ss after -i: frame-accurate
            out = subprocess.run(cmd, capture_output=True)
            buf = np.frombuffer(out.stdout, np.uint8)
            assert out.returncode == 0 and buf.size == n * H * W * 3, \
                f"ffmpeg could not decode {n} frames at {offset} of {self.path}: {out.stderr.decode()[:200]}"
            self.cache.clear(); self.cache[key] = buf.reshape(n, H, W, 3)
        return self.cache[key]


class _Images:
    """Frames of an `image` feature (PNG / JPEG bytes stored in the parquet rows)."""

    def __init__(self, column):
        self.column = column

    def frames(self, offset, n, H, W):
        import cv2
        out = []
        for v in self.column:
            b = v.get("bytes") if isinstance(v, dict) else v
            im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        return np.stack(out)


class LeRobotEpisode(Episode):
    _video = None; _offset = 0; _hw = (0, 0); _depth = None; _dscale = 1.0; _frames = None

    def rgb(self, t):
        if self._frames is None:
            self._frames = self._video.frames(self._offset, self.T, *self._hw)
        return np.ascontiguousarray(self._frames[int(t)])

    def depth_m(self, t):
        if self._depth is None:
            H, W = self.rgb(t).shape[:2]
            return np.zeros((H, W), np.float32)
        return self._depth[int(t)] * self._dscale

    def close(self):
        self._video = None; self._frames = None


def load_episode(path: str, cfg: dict) -> LeRobotEpisode:
    d = cfg["data"]; m = _meta(cfg)
    e = int(path.split("#")[-1]); ep = next(x for x in m.episodes if x["index"] == e)
    rows = m.rows(ep); T = rows.num_rows
    assert T == ep["length"], f"episode {e}: {T} rows but length {ep['length']}"
    col = lambda k: np.stack([np.asarray(v, np.float32) for v in rows[k].to_pylist()])
    state, action = col(d["state_key"]), col(d["action_key"])
    K = np.repeat(np.asarray(d["camera_intrinsics"], np.float64)[None], T, 0)
    Ex = np.asarray(d["camera_extrinsics"], np.float64)
    if d.get("extrinsic_is_cam2base", False):
        M = Ex.reshape(4, 4); E1 = np.zeros((3, 4)); E1[:, :3] = M[:3, :3].T; E1[:, 3] = -M[:3, :3].T @ M[:3, 3]
    else:
        E1 = Ex.reshape(3, 4)
    S = len(d["slots"])
    clicks = np.full((T, S, 2), np.nan, np.float32)
    if d.get("clicks", {}).get("source") == "tracks":           # GAM.tools.track output
        tp = os.path.join(d["clicks"]["tracks_dir"], f"episode_{e}.npz")
        if os.path.exists(tp):
            clicks[:] = np.asarray(np.load(tp)["clicks"], np.float32)[:T, :S]
        else:
            print(f"[lerobot] WARNING no tracks for episode {e} (run GAM.tools.track first)")
    out = LeRobotEpisode(path=path, T=T, state=state, action=action, K=K.astype(np.float32),
                         E=np.repeat(E1[None], T, 0).astype(np.float32),
                         clicks=clicks, instruction=str(ep["task"]), segments=None)
    H, W = m.features[d["camera"]]["shape"][:2]; out._hw = (int(H), int(W))
    if m.features[d["camera"]]["dtype"] == "video":
        vpath, offset = m.video(ep, d["camera"]); out._video, out._offset = _Video(vpath, m.fps), offset
    else:                                                   # image feature: frames live in the parquet rows
        out._video, out._offset = _Images(rows[d["camera"]].to_pylist()), 0
    if d.get("depth_key"):
        shape = tuple(m.features[d["depth_key"]]["shape"])
        out._depth = np.stack([np.asarray(v, np.float32).reshape(shape) for v in rows[d["depth_key"]].to_pylist()])
        out._dscale = 0.001 if d.get("depth_unit", "m") == "mm" else 1.0
    return out
