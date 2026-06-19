"""
nmpc_controller.py
------------------
Kinematic NMPC for UR5e obstacle avoidance using CasADi + IPOPT.

State:        q   ∈ ℝ⁶   (joint angles)
Control:      q̇   ∈ ℝ⁶   (joint velocities)
Prediction:   q(k+1) = q(k) + q̇(k)·dt

Cost:
  J = Σ_{k=0}^{N-1} [
        w_track  · ‖p_ee(q_k) − p_target‖²
      + w_ctrl   · ‖q̇_k‖²
      + w_smooth · ‖q̇_k − q̇_{k-1}‖²
      + w_obs    · Σ_{i,j} max(0, r_safe_{ij} − d(link_i, obs_j))²
      ]
  + w_terminal · ‖p_ee(q_N) − p_target‖²

Constraints (hard):
  q_min ≤ q_k ≤ q_max
  −q̇_max ≤ q̇_k ≤ q̇_max

Obstacle avoidance is implemented as a soft penalty (obstacle cost term)
for real-time feasibility; hard constraints can be enabled via use_hard_obs_constraints.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import casadi as cs

from config import (
    NMPCConfig, ObstacleConfig,
    UR5E_Q_MIN, UR5E_Q_MAX, UR5E_QDOT_MAX,
    UR5E_LINK_RADII,
)
from ur5e_kinematics import _build_fk_expressions, capsule_point_dist_sq

_DOF = 6


# ---------------------------------------------------------------------------
# NMPC solver class
# ---------------------------------------------------------------------------

class KinematicNMPC:
    """
    Builds the NMPC optimisation problem once and re-solves at each control step.
    Obstacle positions are parameters so re-building is never necessary.
    """

    def __init__(
        self,
        cfg:      NMPCConfig   = None,
        obs_cfg:  ObstacleConfig = None,
        n_obs:    int          = 5,
        use_hard_obs_constraints: bool = False,
    ):
        self.cfg     = cfg     or NMPCConfig()
        self.obs_cfg = obs_cfg or ObstacleConfig()
        self.n_obs   = n_obs
        self.use_hard = use_hard_obs_constraints

        self._build_solver()

        # Warm-start storage
        self._q_pred_warm:  Optional[np.ndarray] = None  # (N+1, 6)
        self._qd_pred_warm: Optional[np.ndarray] = None  # (N,   6)

        self.last_solve_time: float = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_solver(self) -> None:
        N  = self.cfg.N
        dt = self.cfg.dt
        n  = self.n_obs

        # ---- symbolic variables ------------------------------------------
        # Decision variables: [q0, q1, …, qN, qd0, qd1, …, qdN-1]  (flattened)
        q_seq  = cs.MX.sym("q_seq",  _DOF * (N + 1))   # states
        qd_seq = cs.MX.sym("qd_seq", _DOF * N)          # controls

        # Parameters:  [q_init(6), target(3), obs_pos(3*n), obs_r(n)]
        p_init   = cs.MX.sym("p_init",   _DOF)
        p_target = cs.MX.sym("p_target", 3)
        p_obs    = cs.MX.sym("p_obs",    3 * n)   # flattened
        p_r_obs  = cs.MX.sym("p_r_obs",  n)

        params = cs.vertcat(p_init, p_target, p_obs, p_r_obs)

        # ---- unpack q and qd sequences -----------------------------------
        def _q(k): return q_seq[k * _DOF : (k + 1) * _DOF]
        def _qd(k): return qd_seq[k * _DOF : (k + 1) * _DOF]

        # ---- build cost and constraints ----------------------------------
        J          = cs.MX(0)
        g_list     = []          # equality + inequality constraints
        lbg_list   = []
        ubg_list   = []

        w_tr  = self.cfg.w_tracking
        w_ct  = self.cfg.w_ctrl
        w_sm  = self.cfg.w_smooth
        w_ob  = self.cfg.w_obstacle
        r_safe = self.obs_cfg.radius + self.obs_cfg.safety_margin

        # Obstacle positions as 3×n matrix for expression building
        obs_mat = cs.reshape(p_obs, 3, n)
        obs_r   = p_r_obs

        for k in range(N):
            qk  = _q(k)
            qdk = _qd(k)

            # -- FK for ee position
            transforms, joint_pos = _build_fk_expressions(qk)
            ee_k = joint_pos[-1]

            # -- Tracking cost
            diff_ee = ee_k - p_target
            J += w_tr * cs.dot(diff_ee, diff_ee)

            # -- Control effort
            J += w_ct * cs.dot(qdk, qdk)

            # -- Smoothness (penalise change in q̇)
            if k > 0:
                dqd = qdk - _qd(k - 1)
                J  += w_sm * cs.dot(dqd, dqd)

            # -- Obstacle soft penalty
            for i_link in range(_DOF):
                cap_a  = joint_pos[i_link]
                cap_b  = joint_pos[i_link + 1]
                r_link = float(UR5E_LINK_RADII[i_link])
                for j_obs in range(n):
                    p_o  = obs_mat[:, j_obs]
                    r_o  = obs_r[j_obs]
                    d_sq = capsule_point_dist_sq(p_o, cap_a, cap_b)
                    d    = cs.sqrt(d_sq + 1e-8)
                    # clearance = d - r_link - r_o; penalise if < r_safe
                    viol = r_safe - (d - r_link - r_o)
                    J   += w_ob * cs.fmax(0.0, viol) ** 2

                    if self.use_hard:
                        clearance = d - r_link - r_o - r_safe
                        g_list.append(clearance)
                        lbg_list.append(0.0)
                        ubg_list.append(float("inf"))

            # -- Dynamics equality: q(k+1) = q(k) + qd(k)*dt
            q_next_pred = qk + qdk * dt
            q_next_var  = _q(k + 1)
            dyn_eq      = q_next_var - q_next_pred
            g_list.append(dyn_eq)
            lbg_list.extend([0.0] * _DOF)
            ubg_list.extend([0.0] * _DOF)

        # -- Initial state equality
        init_eq = _q(0) - p_init
        g_list.append(init_eq)
        lbg_list.extend([0.0] * _DOF)
        ubg_list.extend([0.0] * _DOF)

        # -- Terminal cost
        transforms_N, joint_pos_N = _build_fk_expressions(_q(N))
        ee_N = joint_pos_N[-1]
        diff_N = ee_N - p_target
        J += self.cfg.w_terminal * cs.dot(diff_N, diff_N)

        # ---- bounds on decision variables --------------------------------
        q_min_rep  = np.tile(UR5E_Q_MIN,  N + 1)
        q_max_rep  = np.tile(UR5E_Q_MAX,  N + 1)
        qd_min_rep = np.tile(-UR5E_QDOT_MAX, N)
        qd_max_rep = np.tile( UR5E_QDOT_MAX, N)

        lbx = np.concatenate([q_min_rep,  qd_min_rep])
        ubx = np.concatenate([q_max_rep,  qd_max_rep])

        # ---- assemble NLP ------------------------------------------------
        x_nlp = cs.vertcat(q_seq, qd_seq)
        g_nlp = cs.vertcat(*g_list)

        nlp = {
            "x": x_nlp,
            "f": J,
            "g": g_nlp,
            "p": params,
        }

        opts = {
            "ipopt.max_iter":        self.cfg.ipopt_max_iter,
            "ipopt.tol":             self.cfg.ipopt_tol,
            "ipopt.print_level":     self.cfg.ipopt_print,
            "ipopt.sb":              "yes",
            "print_time":            0,
            "ipopt.warm_start_init_point": "yes",
        }

        self._solver    = cs.nlpsol("nmpc", "ipopt", nlp, opts)
        self._lbx       = lbx
        self._ubx       = ubx
        self._lbg       = np.array(lbg_list)
        self._ubg       = np.array(ubg_list)
        self._n_x       = x_nlp.shape[0]
        self._n_g       = g_nlp.shape[0]
        self._params    = params
        self._N         = N
        self._dt        = dt

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        q_init:      np.ndarray,          # (6,)
        target_pos:  np.ndarray,          # (3,)   world frame
        obs_positions: np.ndarray,        # (n_obs, 3)
        obs_radii:   np.ndarray,          # (n_obs,)
    ) -> Tuple[np.ndarray, dict]:
        """
        Returns:
            q_dot_opt  : (6,)  optimal first control to apply
            info       : dict with solve_time, cost, status, q_pred
        """
        N  = self._N
        dt = self._dt
        n  = self.n_obs

        # Pad / trim obstacles to exactly n_obs
        obs_pos = np.zeros((n, 3))
        obs_r   = np.full(n, 0.001)   # tiny radius for inactive slots
        k_obs   = min(n, len(obs_positions))
        if k_obs > 0:
            obs_pos[:k_obs] = obs_positions[:k_obs]
            obs_r[:k_obs]   = obs_radii[:k_obs]

        # Build parameter vector
        p_val = np.concatenate([
            q_init,
            target_pos,
            obs_pos.flatten(),
            obs_r,
        ])

        # Warm-start
        if self._q_pred_warm is None:
            x0_q  = np.tile(q_init, N + 1)
            x0_qd = np.zeros(_DOF * N)
        else:
            # Shift warm start by one step
            q_warm  = np.vstack([self._q_pred_warm[1:], self._q_pred_warm[-1:]])
            qd_warm = np.vstack([self._qd_pred_warm[1:], self._qd_pred_warm[-1:]])
            x0_q    = q_warm.flatten()
            x0_qd   = qd_warm.flatten()

        x0 = np.concatenate([x0_q, x0_qd])

        t0 = time.perf_counter()
        sol = self._solver(
            x0   = x0,
            lbx  = self._lbx,
            ubx  = self._ubx,
            lbg  = self._lbg,
            ubg  = self._ubg,
            p    = p_val,
        )
        solve_time = time.perf_counter() - t0
        self.last_solve_time = solve_time

        x_opt  = np.array(sol["x"]).flatten()
        cost   = float(sol["f"])
        status = self._solver.stats()["return_status"]

        # Unpack
        q_pred  = x_opt[:_DOF * (N + 1)].reshape(N + 1, _DOF)
        qd_pred = x_opt[_DOF * (N + 1):].reshape(N, _DOF)

        # Store warm start
        self._q_pred_warm  = q_pred
        self._qd_pred_warm = qd_pred

        info = {
            "solve_time": solve_time,
            "cost":       cost,
            "status":     status,
            "q_pred":     q_pred,
            "qd_pred":    qd_pred,
        }

        return qd_pred[0], info

    def reset_warm_start(self) -> None:
        self._q_pred_warm  = None
        self._qd_pred_warm = None
