"""Robot mask for LIBERO without simulator segmentation.

The mask is rendered from what a real robot also has: its own model (meshes + kinematic tree),
the measured joint angles, the camera calibration and the depth image.

  1. forward kinematics from the joint angles -> pose of every robot mesh
  2. rasterise the meshes into the camera -> silhouette + depth of the robot surface
  3. depth test against the observed depth: a pixel whose observed depth is clearly nearer than
     the robot surface is occluded (a held object, the table edge) and is not robot
  4. the gripper-axis marker that robosuite draws in the image is treated as an occluder too

Agreement with the simulator's segmentation render: IoU 0.988-0.993 over four suites.
"""
from __future__ import annotations

import cv2
import numpy as np

PARTS = ("robot0", "gripper0", "mount0")
SS = 4          # supersampling factor of the rasteriser
COV = 0.625     # a pixel is robot if this fraction of its sub-pixels is covered (0.5 + half a sub-pixel of fill bias)


def geom_part(m, g):
    b = int(m.geom_bodyid[g]); nm = m.body_id2name(b) or ""
    while b > 0 and not nm.startswith(PARTS):
        b = int(m.body_parentid[b]); nm = m.body_id2name(b) or ""
    return next((p for p in PARTS if nm.startswith(p)), None)


def sim_meshes(raw, m, part_of):
    """visual geoms of the robot -> {geom_id: [(verts, faces)]} in the geom frame (mesh + primitives)."""
    import trimesh
    out, kinds = {}, {}
    for g in range(raw.ngeom):
        if part_of[g] is None or int(raw.geom_group[g]) != 1:
            continue
        t = int(raw.geom_type[g]); sz = np.asarray(raw.geom_size[g], np.float64)
        if t == 7:   # mesh
            d = int(raw.geom_dataid[g])
            v = raw.mesh_vert[raw.mesh_vertadr[d]:raw.mesh_vertadr[d] + raw.mesh_vertnum[d]]
            f = raw.mesh_face[raw.mesh_faceadr[d]:raw.mesh_faceadr[d] + raw.mesh_facenum[d]]
            out[g] = [(np.asarray(v, np.float64), np.asarray(f, np.int64))]; kinds["mesh"] = kinds.get("mesh", 0) + 1
            continue
        tm = None
        if t == 6: tm = trimesh.creation.box(extents=2 * sz)
        elif t == 5: tm = trimesh.creation.cylinder(radius=sz[0], height=2 * sz[1], sections=24)
        elif t == 2: tm = trimesh.creation.icosphere(subdivisions=2, radius=sz[0])
        elif t == 3: tm = trimesh.creation.capsule(radius=sz[0], height=2 * sz[1])
        nm = {6: "box", 5: "cylinder", 2: "sphere", 3: "capsule"}.get(t, f"type{t}_skipped")
        kinds[nm] = kinds.get(nm, 0) + 1
        if tm is not None:
            out[g] = [(np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64))]
    return out, kinds


def arm_depth(meshes_world, K, E, H, W, nbin=512, ss=None, cov=None):
    """Approximate z-buffer of a set of meshes: triangles are binned by camera depth and painted far -> near
    with cv2.fillPoly (1/16-pixel precision) at `ss` times the image resolution, inside the bounding box of
    the projection only. Downsampling: a pixel is covered if >= `cov` of its sub-pixels are; its depth is the
    nearest covered sub-pixel. Returns (mask (H,W) bool, depth (H,W) float32, inf where empty)."""
    ss = SS if ss is None else ss; cov = COV if cov is None else cov
    K = np.asarray(K, np.float64); E = np.asarray(E, np.float64).reshape(3, 4)
    T, Z = [], []
    for v, f in meshes_world:
        Pc = v @ E[:, :3].T + E[:, 3]; z = Pc[:, 2]
        # continuous pixel coordinates (pixel i spans [i, i+1)) -> cv2 integer grid (integer = pixel centre)
        uv = np.stack([(K[0, 0] * Pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]) * ss - 0.5,
                       (K[1, 1] * Pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]) * ss - 0.5], 1)
        ok = (z > 0.03)[f].all(1); T.append(uv[f[ok]]); Z.append(z[f[ok]].min(1))
    mask = np.zeros((H, W), bool); depth = np.full((H, W), np.inf, np.float32)
    if not T or not sum(len(t) for t in T):
        return mask, depth
    T = np.concatenate(T); Z = np.concatenate(Z)
    lo, hi = T.reshape(-1, 2).min(0), T.reshape(-1, 2).max(0)
    vis = (T[:, :, 0].max(1) >= 0) & (T[:, :, 0].min(1) < W * ss) & (T[:, :, 1].max(1) >= 0) & (T[:, :, 1].min(1) < H * ss)
    T, Z = T[vis], Z[vis]
    if not len(T):
        return mask, depth
    # region of interest, in whole image pixels
    x0 = int(max(0, np.floor(T[:, :, 0].min() / ss))); x1 = int(min(W, np.ceil(T[:, :, 0].max() / ss) + 1))
    y0 = int(max(0, np.floor(T[:, :, 1].min() / ss))); y1 = int(min(H, np.ceil(T[:, :, 1].max() / ss) + 1))
    if x1 <= x0 or y1 <= y0:
        return mask, depth
    w, h = (x1 - x0) * ss, (y1 - y0) * ss
    Ti = np.round((np.clip(T, -4000 * ss, 4000 * ss) - [x0 * ss, y0 * ss]) * 16).astype(np.int32)
    hi_res = np.full((h, w), np.inf, np.float32)
    edges = np.linspace(Z.min(), Z.max() + 1e-6, nbin + 1); b = np.clip(np.digitize(Z, edges) - 1, 0, nbin - 1)
    order = np.argsort(-b, kind="stable"); Ti, b = Ti[order], b[order]
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]]); ends = np.r_[starts[1:], len(b)]
    for s_, e_ in zip(starts, ends):                                   # far -> near
        cv2.fillPoly(hi_res, list(Ti[s_:e_]), float(edges[b[s_]]), shift=4)
    fin = np.isfinite(hi_res)
    # 3x3 nearest: seals the cracks between small triangles that would expose back faces
    hi_res = np.where(fin, cv2.erode(np.where(fin, hi_res, 1e9).astype(np.float32), np.ones((3, 3), np.uint8)), np.inf)
    if ss == 1:
        mask[y0:y1, x0:x1] = fin; depth[y0:y1, x0:x1] = hi_res
        return mask, depth
    fb = fin.reshape(y1 - y0, ss, x1 - x0, ss)
    mk = fb.mean((1, 3)) >= cov
    z = hi_res.reshape(y1 - y0, ss, x1 - x0, ss).min((1, 3))
    mask[y0:y1, x0:x1] = mk; depth[y0:y1, x0:x1] = np.where(mk, z, np.inf)
    return mask, depth


def depth_test(arm, zobs, tol=0.02):
    """A pixel is robot if the robot surface projects there and the observed depth is not more than
    `tol` nearer than that surface (nearer = something in front of the robot: a held object, the table)."""
    mk, zarm = arm
    return mk & (zobs > zarm - tol)


def _long_cyl(r, half, seg=0.04, cap=1.5):
    """A long thin cylinder split into short segments along its axis. The gripper-axis marker is 10 m long;
    as one mesh its triangles would cross behind the camera and be dropped. Only +-cap metres are kept."""
    import trimesh
    h = min(half, cap); n = max(1, int(np.ceil(2 * h / seg))); parts = []
    for k in range(n):
        c = trimesh.creation.cylinder(radius=r, height=2 * h / n, sections=12); c.apply_translation([0, 0, -h + (k + 0.5) * 2 * h / n]); parts.append(c)
    return trimesh.util.concatenate(parts)


def site_meshes(raw, m, names=("gripper0_grip_site_cylinder", "gripper0_grip_site", "gripper0_ft_frame")):
    """Visualisation sites that robosuite draws on the robot (the gripper-axis cylinder, ...) -> {site_id: (verts, faces)}
    in the site frame. They are rendered in the image but are not part of the robot, so they occlude it."""
    import trimesh, mujoco
    out = {}
    for n in names:
        i = mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_SITE, n)
        if i < 0:
            continue
        t = int(raw.site_type[i]); sz = np.asarray(raw.site_size[i], np.float64)
        tm = {6: lambda: trimesh.creation.box(extents=2 * sz), 5: lambda: _long_cyl(sz[0], sz[1]),
              2: lambda: trimesh.creation.icosphere(subdivisions=2, radius=sz[0]), 3: lambda: trimesh.creation.capsule(radius=sz[0], height=2 * sz[1])}.get(t)
        if tm is not None:
            tm = tm(); out[i] = (np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64))
    return out


def final_mask(arm, site, zobs, tol=0.008):
    """robot mask = projected robot surface, minus what the observed depth and the visualisation sites occlude."""
    mk = depth_test(arm, zobs, tol)
    if site is not None:
        mk &= ~(site[0] & (site[1] < arm[1]))
    return mk


class ProjectedRobotMask:
    """mask = ProjectedRobotMask(sim)(obs, depth_m, K, E_world2cam)  ->  (H, W) bool."""

    JOINTS = [f"robot0_joint{i}" for i in range(1, 8)] + ["gripper0_finger_joint1", "gripper0_finger_joint2"]

    def __init__(self, sim):
        import mujoco
        m = sim.model
        self.raw = raw = getattr(m, "_model", m)
        part_of = np.array([geom_part(m, g) for g in range(raw.ngeom)], object)
        self.meshes, _ = sim_meshes(raw, m, part_of)
        self.sites = site_meshes(raw, m)
        self.data = mujoco.MjData(raw)                     # private kinematics buffer: only the robot joints are written
        self.adr = [int(raw.jnt_qposadr[mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in self.JOINTS]
        self._kin = mujoco.mj_kinematics

    def __call__(self, obs, depth_m, K, E):
        d = self.data
        d.qpos[self.adr] = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]).astype(np.float64)
        self._kin(self.raw, d)
        H, W = depth_m.shape[:2]
        robot = [(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g], f) for g, lst in self.meshes.items() for v, f in lst]
        marker = [(v @ d.site_xmat[i].reshape(3, 3).T + d.site_xpos[i], f) for i, (v, f) in self.sites.items()]
        return final_mask(arm_depth(robot, K, E, H, W), arm_depth(marker, K, E, H, W) if marker else None, depth_m)
