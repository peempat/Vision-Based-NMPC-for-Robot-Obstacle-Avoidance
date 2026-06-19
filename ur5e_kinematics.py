"""
ur5e_kinematics.py
------------------
CasADi symbolic forward kinematics, Jacobian, and link-capsule model for UR5e.

Uses standard DH parameters (a, d, alpha from config.py).
All functions build CasADi expressions and compile them once to CasADi Functions
so they can be called quickly inside the NMPC loop.
"""
from __future__ import annotations

import numpy as np
import casadi as cs

from config import (
    UR5E_A, UR5E_D, UR5E_ALPHA,
    UR5E_Q_MIN, UR5E_Q_MAX, UR5E_QDOT_MAX,
    UR5E_LINK_RADII,
)

_DOF = 6


# ---------------------------------------------------------------------------
# DH transform (symbolic)
# ---------------------------------------------------------------------------

def _dh_mat(theta: cs.MX, d: float, a: float, alpha: float) -> cs.MX:
    """Standard DH 4×4 homogeneous transform (CasADi symbolic)."""
    ct = cs.cos(theta)
    st = cs.sin(theta)
    ca = float(np.cos(alpha))
    sa = float(np.sin(alpha))
    row0 = cs.horzcat(ct,     -st * ca,   st * sa,  a * ct)
    row1 = cs.horzcat(st,      ct * ca,  -ct * sa,  a * st)
    row2 = cs.horzcat(0.0,     sa,        ca,        d     )
    row3 = cs.horzcat(0.0,     0.0,       0.0,       1.0   )
    return cs.vertcat(row0, row1, row2, row3)


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def _build_fk_expressions(q: cs.MX):
    """
    Returns:
        transforms  : list of 6 cumulative 4×4 transforms T_0_i
        joint_positions : list of 7 position vectors (p0=origin, p1…p6)
    """
    T = cs.MX.eye(4)
    transforms = []
    joint_pos  = [cs.MX.zeros(3)]        # p0 = robot base origin

    for i in range(_DOF):
        Ti = _dh_mat(q[i], UR5E_D[i], UR5E_A[i], UR5E_ALPHA[i])
        T  = T @ Ti
        transforms.append(T)
        joint_pos.append(T[:3, 3])

    return transforms, joint_pos


def build_fk_function() -> cs.Function:
    """
    CasADi Function: q (6,) → ee_pos (3,)
    Compiles the symbolic graph once; call repeatedly without overhead.
    """
    q = cs.MX.sym("q", _DOF)
    transforms, joint_pos = _build_fk_expressions(q)
    ee_pos = joint_pos[-1]
    ee_rot = transforms[-1][:3, :3]
    return cs.Function(
        "fk",
        [q],
        [ee_pos, cs.reshape(ee_rot, 9, 1)],
        ["q"],
        ["ee_pos", "ee_rot_flat"],
    )


def build_link_positions_function() -> cs.Function:
    """
    CasADi Function: q (6,) → joint_positions (7×3), column per joint.
    Returns positions p0…p6 (p0 = base, p6 = ee).
    """
    q = cs.MX.sym("q", _DOF)
    _, joint_pos = _build_fk_expressions(q)
    positions = cs.horzcat(*joint_pos)   # shape 3×7
    return cs.Function(
        "link_positions",
        [q],
        [positions],
        ["q"],
        ["positions"],
    )


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

def build_jacobian_function() -> cs.Function:
    """
    CasADi Function: q (6,) → J (3×6) position Jacobian of end-effector.
    """
    q   = cs.MX.sym("q", _DOF)
    transforms, joint_pos = _build_fk_expressions(q)
    ee_pos = joint_pos[-1]
    J  = cs.jacobian(ee_pos, q)
    return cs.Function("jacobian", [q], [J], ["q"], ["J"])


# ---------------------------------------------------------------------------
# Capsule model (symbolic, for NMPC)
# ---------------------------------------------------------------------------

def capsule_point_dist_sq(p_obs: cs.MX, cap_a: cs.MX, cap_b: cs.MX) -> cs.MX:
    """
    Squared distance from point p_obs (3,) to line segment cap_a→cap_b (3, each).
    Uses a smooth clamp so the expression is differentiable everywhere.
    """
    ab    = cap_b - cap_a
    ap    = p_obs - cap_a
    ab_sq = cs.dot(ab, ab) + 1e-8
    t     = cs.dot(ap, ab) / ab_sq
    t_c   = cs.fmax(0.0, cs.fmin(1.0, t))
    closest = cap_a + t_c * ab
    diff  = p_obs - closest
    return cs.dot(diff, diff)


def build_capsule_distances_function(n_obs: int) -> cs.Function:
    """
    CasADi Function:
        q      (6,)
        obs    (3, n_obs)  — obstacle positions (sphere centres in world frame)
        r_obs  (n_obs,)    — obstacle sphere radii
    Returns:
        min_dists (6, n_obs) — clearance distance for each (link, obstacle) pair
                               positive = clear, negative = penetrating
    """
    q      = cs.MX.sym("q",     _DOF)
    obs    = cs.MX.sym("obs",   3, n_obs)
    r_obs  = cs.MX.sym("r_obs", n_obs)

    _, joint_pos = _build_fk_expressions(q)

    dist_exprs = []
    for i_link in range(_DOF):
        cap_a = joint_pos[i_link]
        cap_b = joint_pos[i_link + 1]
        r_link = float(UR5E_LINK_RADII[i_link])
        for j_obs in range(n_obs):
            p_o   = obs[:, j_obs]
            r_o   = r_obs[j_obs]
            d_sq  = capsule_point_dist_sq(p_o, cap_a, cap_b)
            d_raw = cs.sqrt(d_sq + 1e-8)
            clearance = d_raw - r_link - r_o
            dist_exprs.append(clearance)

    dists = cs.vertcat(*dist_exprs)   # shape (6*n_obs,)
    dists_mat = cs.reshape(dists, _DOF, n_obs)

    return cs.Function(
        "capsule_distances",
        [q, obs, r_obs],
        [dists_mat],
        ["q", "obs", "r_obs"],
        ["dists"],
    )


# ---------------------------------------------------------------------------
# Numpy helpers (for non-symbolic use in baselines / evaluation)
# ---------------------------------------------------------------------------

def fk_numpy(q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Numerical FK using numpy.
    Returns: (ee_pos [3,], joint_positions list of 7 arrays shape [3,])
    """
    T = np.eye(4)
    joint_pos = [np.zeros(3)]
    for i in range(_DOF):
        theta = q[i]
        d     = UR5E_D[i]
        a     = UR5E_A[i]
        alpha = UR5E_ALPHA[i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        Ti = np.array([
            [ct,  -st*ca,   st*sa,  a*ct],
            [st,   ct*ca,  -ct*sa,  a*st],
            [0.0,  sa,      ca,     d   ],
            [0.0,  0.0,     0.0,    1.0 ],
        ])
        T = T @ Ti
        joint_pos.append(T[:3, 3].copy())
    return joint_pos[-1], joint_pos


def jacobian_numpy(q: np.ndarray) -> np.ndarray:
    """Numerical 3×6 position Jacobian using finite differences."""
    eps = 1e-6
    J = np.zeros((3, _DOF))
    p0, _ = fk_numpy(q)
    for i in range(_DOF):
        dq = np.zeros(_DOF)
        dq[i] = eps
        pi, _ = fk_numpy(q + dq)
        J[:, i] = (pi - p0) / eps
    return J


def capsule_point_dist_numpy(
    p_obs: np.ndarray,
    cap_a: np.ndarray,
    cap_b: np.ndarray,
    r_link: float,
    r_obs: float,
) -> float:
    """Clearance distance (positive = safe) between point obstacle and capsule."""
    ab = cap_b - cap_a
    ap = p_obs - cap_a
    ab_sq = np.dot(ab, ab) + 1e-8
    t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)
    closest = cap_a + t * ab
    d = np.linalg.norm(p_obs - closest)
    return d - r_link - r_obs


def min_clearance_numpy(
    q: np.ndarray,
    obstacles: list[tuple[np.ndarray, float]],
) -> float:
    """
    Minimum clearance over all (link, obstacle) pairs.
    obstacles: list of (position [3,], radius float)
    """
    _, joint_pos = fk_numpy(q)
    min_d = float("inf")
    for i_link in range(_DOF):
        cap_a  = joint_pos[i_link]
        cap_b  = joint_pos[i_link + 1]
        r_link = UR5E_LINK_RADII[i_link]
        for p_obs, r_obs in obstacles:
            d = capsule_point_dist_numpy(p_obs, cap_a, cap_b, r_link, r_obs)
            if d < min_d:
                min_d = d
    return min_d
