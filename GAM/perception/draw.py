"""Draw the per-slot 3D boxes the policy consumes onto a camera frame (evaluation videos)."""
from __future__ import annotations

import numpy as np

SLOT_COLORS = [(66, 135, 245), (52, 199, 89), (255, 159, 10), (191, 90, 242), (255, 69, 58), (100, 210, 255)]   # RGB
_SGN = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], np.float32)     # as in geometry.box2d_from_obb
_EDGES = [(i, i ^ m) for i in range(8) for m in (1, 2, 4) if i < i ^ m]


def rot6d_to_R(r6) -> np.ndarray:
    """6-d rotation (first two box axes) -> orthonormal 3x3 with the axes as columns."""
    a = np.asarray(r6, np.float64)[:3]; b = np.asarray(r6, np.float64)[3:6]
    a = a / max(np.linalg.norm(a), 1e-8); b = b - a * (a @ b); b = b / max(np.linalg.norm(b), 1e-8)
    return np.stack([a, b, np.cross(a, b)], 1)


def box_corners(center, extents, R) -> np.ndarray:
    """(8,3) world corners of an oriented box."""
    return np.asarray(center, np.float32) + (_SGN * np.asarray(extents, np.float32) / 2) @ np.asarray(R, np.float32).reshape(3, 3).T


def project(pts, K, E):
    """world points -> (n,2) pixels and (n,) in-front flags; E = world->camera 3x4 (OpenCV), K 3x3."""
    E = np.asarray(E, np.float64).reshape(3, 4); K = np.asarray(K, np.float64).reshape(3, 3)
    pc = np.asarray(pts, np.float64) @ E[:, :3].T + E[:, 3]
    z = np.clip(pc[:, 2], 1e-4, None)
    uv = (pc[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv, pc[:, 2] > 0.01


def draw_box3d(img, center, extents, R, K, E, color=(66, 135, 245), thickness=2):
    """Wireframe of one oriented 3D box on `img` (modified in place and returned)."""
    import cv2
    uv, ok = project(box_corners(center, extents, R), K, E)
    if ok.sum() < 8:
        return img
    p = np.round(uv).astype(int)
    for i, j in _EDGES:
        cv2.line(img, tuple(p[i]), tuple(p[j]), color, thickness, cv2.LINE_AA)
    return img


def draw_slots(img, xyz, boxgeo, K, E, names=None, thickness=2):   # names kept for callers; boxes carry no text
    """All slots of one frame: xyz (S,3) world centres, boxgeo (S,9) = extents + rot6d (empty slot = zeros)."""
    for k in range(len(xyz)):
        ext = np.asarray(boxgeo[k][:3], np.float32)
        if ext.max() <= 0:
            continue
        draw_box3d(img, xyz[k], ext, rot6d_to_R(boxgeo[k][3:9]), K, E, SLOT_COLORS[k % len(SLOT_COLORS)], thickness=thickness)
    return img


def draw_boxes(img, boxes, K, E, slots=None, thickness=2):
    """Boxes as GAMRunner returns them ([{slot, p, extents, R}, ...]); colour by slot index."""
    for b in boxes or []:
        k = slots.index(b["slot"]) if slots and b["slot"] in slots else 0
        draw_box3d(img, b["p"], b["extents"], np.asarray(b["R"], np.float32).reshape(3, 3), K, E, SLOT_COLORS[k % len(SLOT_COLORS)], thickness)
    return img
