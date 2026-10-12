"""Forward kinematics of the Franka Panda (standard franka_description joint origins).

Only what the policy needs: the TCP position in the robot base frame from the 7 arm joint
angles, used to express object geometry relative to the end effector (`train.pcd_ref: ee`).
"""
from __future__ import annotations

import numpy as np

# (xyz, rpy) of joint i's origin in its parent link; every joint rotates about its own z.
_ORIGINS = [
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-np.pi / 2, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (np.pi / 2, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-np.pi / 2, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
]
_FLANGE_Z = 0.107            # link7 -> link8 (flange)
_HAND_YAW = -np.pi / 4       # link8 -> panda_hand
TCP_Z = 0.1034               # panda_hand -> fingertip centre


def _T(xyz, rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def _Rz(q):
    T = np.eye(4)
    T[:2, :2] = [[np.cos(q), -np.sin(q)], [np.sin(q), np.cos(q)]]
    return T


def hand_pose(q7) -> np.ndarray:
    """4x4 pose of `panda_hand` in the base (panda_link0) frame."""
    T = np.eye(4)
    for (xyz, rpy), q in zip(_ORIGINS, np.asarray(q7, np.float64)[:7]):
        T = T @ _T(xyz, rpy) @ _Rz(float(q))
    return T @ _T((0.0, 0.0, _FLANGE_Z), (0.0, 0.0, 0.0)) @ _T((0.0, 0.0, 0.0), (0.0, 0.0, _HAND_YAW))


def tcp_position(q7) -> np.ndarray:
    """(3,) fingertip-centre position in the base frame."""
    return (hand_pose(q7) @ np.array([0.0, 0.0, TCP_Z, 1.0]))[:3]
