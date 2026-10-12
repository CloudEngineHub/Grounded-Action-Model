"""Simulator-side helpers for LIBERO: camera calibration, metric depth and the robot base pose.
These are the quantities a real setup gets from calibration, no object or scene ground truth."""
import math

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0])   # mujoco / OpenGL camera axes -> OpenCV camera axes


def cam_K(sim, cam_name, W, H):
    cid = sim.model.camera_name2id(cam_name)
    f = (H / 2.0) / math.tan(math.radians(float(sim.model.cam_fovy[cid]) / 2.0))
    return np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]], np.float64)


def cam_ext_cv(sim, cam_name):
    """Returns (R_cv, campos, E_t). R_cv: cam->world rotation (OpenCV convention),
    campos: cam origin in world. E_t: 3x4 world->cam [R_w2c | t_w2c]."""
    cid = sim.model.camera_name2id(cam_name)
    campos = np.asarray(sim.data.cam_xpos[cid], np.float64).copy()
    R_cv = np.asarray(sim.data.cam_xmat[cid], np.float64).reshape(3, 3) @ FLIP
    R_w2c = R_cv.T
    t_w2c = -R_w2c @ campos
    E_t = np.concatenate([R_w2c, t_w2c[:, None]], axis=1)
    return R_cv, campos, E_t


def real_depth(sim, d_norm):
    """mujoco normalized z-buffer -> meters (== robosuite get_real_depth_map)."""
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    return near / (1.0 - d_norm * (1.0 - near / far))


def base_T(sim):
    """Robot base pose in world -> (baseT 4x4, baseTi, pose7 [pos3, quat_wxyz4])."""
    from scipy.spatial.transform import Rotation as SciR
    bid = None
    for nm in ("robot0_base", "robot0_link0", "robot0_fixed_base_link"):
        try:
            bid = sim.model.body_name2id(nm); break
        except Exception:
            continue
    assert bid is not None, "robot base body not found"
    pos = np.asarray(sim.data.body_xpos[bid], np.float64).copy()
    R = np.asarray(sim.data.body_xmat[bid], np.float64).reshape(3, 3).copy()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    q_xyzw = SciR.from_matrix(R).as_quat()
    pose7 = np.concatenate([pos, [q_xyzw[3]], q_xyzw[:3]]).astype(np.float32)
    return T, np.linalg.inv(T), pose7
