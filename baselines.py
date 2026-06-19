"""
baselines.py
------------
Baseline motion planners for comparison with Kinematic NMPC.

    1. Artificial Potential Field (APF)
       — runs at control frequency, reactive, no horizon.
       — prone to local minima but very fast.

    2. RRTConnect
       — offline planner in joint space, produces a joint trajectory.
       — guarantees path (probabilistically complete) but ignores dynamics.

Both use the same YOLO+ArUco obstacle positions as the NMPC.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config import (
    BaselineConfig, ObstacleConfig,
    UR5E_Q_MIN, UR5E_Q_MAX, UR5E_QDOT_MAX, UR5E_LINK_RADII,
)
from ur5e_kinematics import (
    fk_numpy, jacobian_numpy,
    capsule_point_dist_numpy, min_clearance_numpy,
)

_DOF = 6


# ===========================================================================
# 1.  Artificial Potential Field
# ===========================================================================

class APFController:
    """
    Gradient-descent in end-effector space with Jacobian pseudo-inverse.

    Attractive potential:  U_att = ½ α ‖p_ee − p_goal‖²
    Repulsive potential:   U_rep = ½ β (1/d − 1/d0)²  if d < d0
    Joint velocity:        q̇ = J† (−∇U_att − ∇U_rep)
    """

    def __init__(
        self,
        cfg:     BaselineConfig  = None,
        obs_cfg: ObstacleConfig  = None,
    ):
        self.cfg     = cfg     or BaselineConfig()
        self.obs_cfg = obs_cfg or ObstacleConfig()

    def compute(
        self,
        q:            np.ndarray,              # (6,)
        target_pos:   np.ndarray,              # (3,)
        obstacles:    List[Tuple[np.ndarray, float]],  # (pos [3], radius)
    ) -> np.ndarray:
        """Returns q_dot (6,)."""
        ee_pos, _ = fk_numpy(q)
        J = jacobian_numpy(q)   # 3×6

        # Attractive: F_att = −α * (p_ee − p_goal)
        F_att = -self.cfg.apf_alpha * (ee_pos - target_pos)

        # Repulsive: sum over obstacles
        F_rep = np.zeros(3)
        for p_obs, r_obs in obstacles:
            r_safe = r_obs + self.obs_cfg.radius + self.obs_cfg.safety_margin
            d_obs  = float(np.linalg.norm(ee_pos - p_obs))
            d_eff  = max(d_obs - r_obs, 1e-4)
            if d_eff < self.cfg.apf_d0:
                mag = self.cfg.apf_beta * (1.0 / d_eff - 1.0 / self.cfg.apf_d0) / (d_eff ** 2)
                direction = (ee_pos - p_obs) / (d_obs + 1e-8)
                F_rep += mag * direction

        F_total = F_att + F_rep

        # Pseudo-inverse Jacobian
        J_pinv = np.linalg.pinv(J)
        q_dot  = J_pinv @ F_total
        q_dot  = np.clip(q_dot, -UR5E_QDOT_MAX, UR5E_QDOT_MAX)
        return q_dot


# ===========================================================================
# 2.  RRTConnect
# ===========================================================================

@dataclass
class _Node:
    q:      np.ndarray
    parent: Optional["_Node"] = None


def _random_q() -> np.ndarray:
    return np.random.uniform(UR5E_Q_MIN, UR5E_Q_MAX)


def _nearest(tree: List[_Node], q: np.ndarray) -> _Node:
    dists = [np.linalg.norm(n.q - q) for n in tree]
    return tree[int(np.argmin(dists))]


def _steer(q_near: np.ndarray, q_rand: np.ndarray, step: float) -> np.ndarray:
    diff = q_rand - q_near
    dist = np.linalg.norm(diff)
    if dist < step:
        return q_rand.copy()
    return q_near + step * diff / dist


def _is_collision_free(
    q:          np.ndarray,
    obstacles:  List[Tuple[np.ndarray, float]],
    safety:     float,
) -> bool:
    if not obstacles:
        return True
    _, joint_pos = fk_numpy(q)
    for i_link in range(_DOF):
        cap_a  = joint_pos[i_link]
        cap_b  = joint_pos[i_link + 1]
        r_link = UR5E_LINK_RADII[i_link]
        for p_obs, r_obs in obstacles:
            d = capsule_point_dist_numpy(p_obs, cap_a, cap_b, r_link, r_obs)
            if d < safety:
                return False
    return True


def _check_path(
    q_from:    np.ndarray,
    q_to:      np.ndarray,
    obstacles: List[Tuple[np.ndarray, float]],
    safety:    float,
    n_check:   int = 5,
) -> bool:
    for i in range(n_check + 1):
        q_mid = q_from + (i / n_check) * (q_to - q_from)
        if not _is_collision_free(q_mid, obstacles, safety):
            return False
    return True


def _extract_path(node: _Node) -> List[np.ndarray]:
    path = []
    while node is not None:
        path.append(node.q)
        node = node.parent
    return list(reversed(path))


def _connect(
    tree:      List[_Node],
    q_target:  np.ndarray,
    step:      float,
    obstacles: List[Tuple[np.ndarray, float]],
    safety:    float,
) -> Tuple[bool, Optional[_Node]]:
    """
    Greedy extension of tree toward q_target.
    Returns (reached, last_node).
    """
    while True:
        q_near = _nearest(tree, q_target).q
        q_new  = _steer(q_near, q_target, step)
        if not _check_path(q_near, q_new, obstacles, safety):
            return False, None
        node = _Node(q=q_new, parent=_nearest(tree, q_near))
        tree.append(node)
        if np.linalg.norm(q_new - q_target) < 1e-3:
            return True, node
        if np.linalg.norm(q_new - q_near) < 1e-6:
            return False, None


class RRTConnectPlanner:
    """
    Bidirectional RRTConnect planner in joint space.
    Returns a list of joint configurations forming the path,
    or None if planning fails within max_iter.
    """

    def __init__(
        self,
        cfg:     BaselineConfig  = None,
        obs_cfg: ObstacleConfig  = None,
    ):
        self.cfg     = cfg     or BaselineConfig()
        self.obs_cfg = obs_cfg or ObstacleConfig()

    def plan(
        self,
        q_start:     np.ndarray,
        q_goal:      np.ndarray,
        obstacles:   List[Tuple[np.ndarray, float]],
    ) -> Tuple[Optional[List[np.ndarray]], float]:
        """
        Returns (path, solve_time).
        path is None if planning fails.
        """
        safety = 0.005   # 5 mm physical buffer; safety_margin is for NMPC penalty, not path planning
        step   = self.cfg.rrt_step_size
        t0     = time.perf_counter()

        tree_a = [_Node(q=q_start.copy())]
        tree_b = [_Node(q=q_goal.copy())]

        for _ in range(self.cfg.rrt_max_iter):
            q_rand = _random_q()

            # Extend tree_a toward q_rand
            q_near_a  = _nearest(tree_a, q_rand).q
            q_new_a   = _steer(q_near_a, q_rand, step)
            if _check_path(q_near_a, q_new_a, obstacles, safety):
                node_a = _Node(q=q_new_a, parent=_nearest(tree_a, q_near_a))
                tree_a.append(node_a)

                # Connect tree_b toward q_new_a
                reached, node_b = _connect(tree_b, q_new_a, step, obstacles, safety)
                if reached and node_b is not None:
                    path_a = _extract_path(node_a)
                    path_b = _extract_path(node_b)
                    path   = path_a + list(reversed(path_b))
                    solve_time = time.perf_counter() - t0
                    return path, solve_time

            # Swap trees each iteration (bidirectional)
            tree_a, tree_b = tree_b, tree_a

        solve_time = time.perf_counter() - t0
        return None, solve_time

    def path_to_control(
        self,
        path:  List[np.ndarray],
        dt:    float = 0.05,
    ) -> List[np.ndarray]:
        """
        Convert a joint-space path to a sequence of joint velocities.
        Returns list of q_dot arrays.
        """
        controls = []
        for i in range(len(path) - 1):
            dq   = path[i + 1] - path[i]
            q_dot = np.clip(dq / dt, -UR5E_QDOT_MAX, UR5E_QDOT_MAX)
            controls.append(q_dot)
        return controls


# ===========================================================================
# Helper: run APF for one episode and collect metrics
# ===========================================================================

def run_apf_episode(
    q_start:      np.ndarray,
    target_pos:   np.ndarray,
    obstacle_fn,                   # callable() → List[Tuple[pos, radius]]
    apf_cfg:      BaselineConfig = None,
    obs_cfg:      ObstacleConfig = None,
    max_steps:    int = 600,
    dt:           float = 0.05,
    goal_tol:     float = 0.05,
) -> dict:
    """Pure Python APF episode (no MuJoCo) — useful for quick benchmarking."""
    from evaluation import EpisodeEvaluator
    apf  = APFController(apf_cfg, obs_cfg)
    eval = EpisodeEvaluator()
    q    = q_start.copy()

    for step in range(max_steps):
        t0         = time.perf_counter()
        obstacles  = obstacle_fn()
        q_dot      = apf.compute(q, target_pos, obstacles)
        solve_time = time.perf_counter() - t0

        q += q_dot * dt
        q  = np.clip(q, UR5E_Q_MIN, UR5E_Q_MAX)

        ee_pos, _ = fk_numpy(q)
        clearance = min_clearance_numpy(q, obstacles) if obstacles else float("inf")

        eval.record(
            sim_time     = step * dt,
            ee_pos       = ee_pos,
            target_pos   = target_pos,
            q_dot        = q_dot,
            clearance    = clearance,
            solve_time   = solve_time,
            solver_status= "APF",
        )

        if np.linalg.norm(ee_pos - target_pos) < goal_tol:
            break

    return eval.summary()


# ===========================================================================
# 3.  MuJoCo-compatible wrappers (same solve() interface as KinematicNMPC)
# ===========================================================================

class APFControllerMuJoCo:
    """
    Wraps APFController to match KinematicNMPC.solve() so it can be
    passed directly to UR5eSimulation.run().
    """

    def __init__(self, cfg: BaselineConfig = None, obs_cfg: ObstacleConfig = None):
        self._apf    = APFController(cfg, obs_cfg)
        self.obs_cfg = obs_cfg or ObstacleConfig()

    def solve(
        self,
        q_init:        np.ndarray,
        target_pos:    np.ndarray,
        obs_positions: np.ndarray,   # (N, 3)
        obs_radii:     np.ndarray,   # (N,)
    ):
        import time
        obstacles = [
            (obs_positions[i], float(obs_radii[i]))
            for i in range(len(obs_radii))
        ] if len(obs_radii) > 0 else []
        t0    = time.perf_counter()
        q_dot = self._apf.compute(q_init, target_pos, obstacles)
        return q_dot, {
            "status":     "APF",
            "solve_time": time.perf_counter() - t0,
        }

    def reset_warm_start(self) -> None:
        pass


class RRTPathExecutor:
    """
    Executes a pre-planned RRT joint-space path as a velocity controller.
    Matches KinematicNMPC.solve() interface for use with UR5eSimulation.run().

    At each control tick it steps forward one node in the path and returns
    the velocity needed to reach that node within ctrl_dt.
    """

    def __init__(
        self,
        path:    List[np.ndarray],   # list of q configs from RRTConnectPlanner.plan()
        ctrl_dt: float = 0.2,        # must match sim_cfg control period
    ):
        self._path    = path or []
        self._idx     = 0
        self._ctrl_dt = ctrl_dt

    def solve(
        self,
        q_init:        np.ndarray,
        target_pos:    np.ndarray,
        obs_positions: np.ndarray,
        obs_radii:     np.ndarray,
    ):
        import time
        t0 = time.perf_counter()
        if self._idx < len(self._path) - 1:
            self._idx += 1
            q_next = self._path[self._idx]
            q_dot  = (q_next - q_init) / self._ctrl_dt
            q_dot  = np.clip(q_dot, -UR5E_QDOT_MAX, UR5E_QDOT_MAX)
        else:
            q_dot = np.zeros(_DOF)
        return q_dot, {
            "status":     "RRT",
            "solve_time": time.perf_counter() - t0,
        }

    def reset_warm_start(self) -> None:
        self._idx = 0
