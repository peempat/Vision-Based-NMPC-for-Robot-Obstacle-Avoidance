"""
mujoco_sim.py
-------------
MuJoCo simulation interface for the UR5e obstacle-avoidance system.

Provides:
    get_robot_state()         — read joint positions and velocities
    compute_forward_kinematics() — link positions via MuJoCo engine (ground truth)
    compute_link_capsules()   — (p1, p2, r) per link from MuJoCo kinematics
    update_obstacle_positions() — move mocap obstacle bodies
    apply_control()           — set actuator position targets (kinematic position servo)
    simulation_step()         — advance physics
    update_trajectory_vis()   — update predicted-path sites
    main_loop()               — complete simulation loop (called by main.py)
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from config import SimConfig, NMPCConfig, ObstacleConfig, UR5E_HOME_Q, UR5E_LINK_RADII
from ur5e_kinematics import fk_numpy

_DOF     = 6
_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_LINK_BODY_NAMES = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]
_MAX_OBS = 5   # must match scene_obstacle_avoidance.xml
_TRAJ_SITES = 11  # traj_0 … traj_10


class UR5eSimulation:
    """
    Wraps a MuJoCo model + data for the UR5e obstacle-avoidance scene.
    """

    def __init__(
        self,
        sim_cfg:     SimConfig      = None,
        nmpc_cfg:    NMPCConfig     = None,
        obs_cfg:     ObstacleConfig = None,
        scene_dir:   str            = None,
    ):
        self.sim_cfg  = sim_cfg  or SimConfig()
        self.nmpc_cfg = nmpc_cfg or NMPCConfig()
        self.obs_cfg  = obs_cfg  or ObstacleConfig()

        if scene_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            scene_dir = os.path.join(
                here,
                self.sim_cfg.model_dir,
            )
        self.scene_dir = scene_dir

        xml_path = os.path.join(scene_dir, self.sim_cfg.scene_xml)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # Cache joint and body IDs
        self._joint_ids = [
            self.model.joint(name).id for name in _JOINT_NAMES
        ]
        self._actuator_ids = list(range(_DOF))   # actuators are in order

        self._obs_body_ids = []
        for k in range(_MAX_OBS):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{k}")
            self._obs_body_ids.append(bid)

        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body"
        )

        self._traj_site_ids = []
        for k in range(_TRAJ_SITES):
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"traj_{k}"
            )
            self._traj_site_ids.append(sid)

        # Cache link body IDs for MuJoCo FK ground truth
        self._link_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in _LINK_BODY_NAMES
        ]

        # EE site
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

        # Initialise to home configuration
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, q: Optional[np.ndarray] = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q0 = q if q is not None else UR5E_HOME_Q.copy()
        for i, jid in enumerate(self._joint_ids):
            self.data.qpos[self.model.joint(jid).qposadr[0]] = q0[i]
        self.data.ctrl[:] = q0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_robot_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (q [6,], q_dot [6,]) from MuJoCo data."""
        q     = np.array([
            float(self.data.qpos[self.model.joint(jid).qposadr[0]])
            for jid in self._joint_ids
        ])
        q_dot = np.array([
            float(self.data.qvel[self.model.joint(jid).dofadr[0]])
            for jid in self._joint_ids
        ])
        return q, q_dot

    # ------------------------------------------------------------------
    # Kinematics via MuJoCo engine (ground truth)
    # ------------------------------------------------------------------

    def compute_forward_kinematics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (ee_pos [3,], ee_rot [3,3]) from MuJoCo body kinematics.
        More accurate than DH FK because it uses the actual model geometry.
        """
        mujoco.mj_forward(self.model, self.data)
        if self._ee_site_id >= 0:
            ee_pos = self.data.site_xpos[self._ee_site_id].copy()
            ee_rot = self.data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
        else:
            # Fall back to last link body
            ee_pos = self.data.xpos[self._link_body_ids[-1]].copy()
            ee_rot = self.data.xmat[self._link_body_ids[-1]].reshape(3, 3).copy()
        return ee_pos, ee_rot

    def compute_link_capsules(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Returns list of (p_start, p_end, radius) capsules for each link,
        using MuJoCo body positions (ground truth).
        Shape: 6 entries, one per UR5e link.
        """
        mujoco.mj_forward(self.model, self.data)
        body_positions = [
            self.data.xpos[bid].copy() for bid in self._link_body_ids
        ]
        # Base position (world origin or explicit base body position)
        base_pos = self.data.xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        ].copy()

        all_pts = [base_pos] + body_positions
        capsules = []
        for i in range(_DOF):
            capsules.append((all_pts[i], all_pts[i + 1], UR5E_LINK_RADII[i]))
        return capsules

    # ------------------------------------------------------------------
    # Obstacle visualization
    # ------------------------------------------------------------------

    def update_obstacle_positions(
        self,
        positions:  List[np.ndarray],   # list of [x, y, z]
        park_far:   bool = True,
    ) -> None:
        """Move mocap obstacle bodies to given world positions."""
        for k in range(_MAX_OBS):
            bid = self._obs_body_ids[k]
            if bid < 0:
                continue
            mocap_id = self.model.body_mocapid[bid]
            if mocap_id < 0:
                continue
            if k < len(positions):
                self.data.mocap_pos[mocap_id] = positions[k]
            elif park_far:
                self.data.mocap_pos[mocap_id] = np.array([10.0, float(k), 0.0])

    def set_target_position(self, pos: np.ndarray) -> None:
        """Move the target sphere to a world position."""
        bid = self._target_body_id
        if bid < 0:
            return
        mocap_id = self.model.body_mocapid[bid]
        if mocap_id >= 0:
            self.data.mocap_pos[mocap_id] = pos

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def apply_control(self, q_cmd: np.ndarray) -> None:
        """
        Set actuator position targets.
        The UR5e actuators in ur5e.xml use gaintype=fixed (position servo).
        """
        self.data.ctrl[:_DOF] = np.clip(q_cmd, -6.3, 6.3)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def simulation_step(self, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------
    # Trajectory visualisation
    # ------------------------------------------------------------------

    def update_trajectory_vis(self, q_pred: np.ndarray) -> None:
        """
        Update predicted NMPC trajectory sites.
        q_pred: (N+1, 6) — predicted joint angles.
        """
        n_steps = min(q_pred.shape[0], _TRAJ_SITES)
        for k in range(n_steps):
            ee_pos, _ = fk_numpy(q_pred[k])
            sid = self._traj_site_ids[k]
            if sid >= 0:
                self.model.site_pos[sid] = ee_pos

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(
        self,
        nmpc_controller,                # KinematicNMPC (or APF/RRT wrapper) instance
        target_pos:    np.ndarray,      # (3,) world position — used as initial target
        obstacle_source=None,           # callable() → List[np.ndarray] (world pos)
        target_source=None,             # callable() → np.ndarray — dynamic target; overrides target_pos each tick
        initial_q:     Optional[np.ndarray] = None,  # override home joint config
        viewer_azimuth:   float = 120,  # MuJoCo viewer camera azimuth [deg]
        viewer_elevation: float = -20,  # MuJoCo viewer camera elevation [deg]
        viewer_distance:  float = 1.8,  # MuJoCo viewer camera distance [m]
        viewer_lookat:    Optional[np.ndarray] = None,  # MuJoCo viewer lookat [x,y,z]
        max_duration:  float = 60.0,    # seconds of active control
        goal_tol:      float = 0.02,    # [m] success threshold
        save_video:    bool  = True,    # capture offscreen frames for GIF
        video_fps:     int   = 10,      # video frame rate
        max_video_frames: int = 500,    # cap memory usage
        headless:      bool  = False,   # no viewer — runs to max_duration then returns
    ) -> Tuple[dict, List[np.ndarray]]:
        """
        Main control loop.

        Returns:
            (results_dict, video_frames)
            results_dict  — from EpisodeEvaluator.summary(), includes "trajectory" key
            video_frames  — list of (H, W, 3) uint8 arrays for GIF saving

        headless=True: runs without a viewer window, completes automatically when
            max_duration is reached or the goal is held for 2 s.  Used by compare mode.
        """
        from evaluation import EpisodeEvaluator
        from ur5e_kinematics import min_clearance_numpy
        evaluator = EpisodeEvaluator()

        _lookat = viewer_lookat if viewer_lookat is not None else np.array([0.3, 0.0, 0.4])

        self.reset(q=initial_q)
        self.set_target_position(target_pos)

        ctrl_dt  = 1.0 / self.sim_cfg.control_freq
        sim_dt   = self.sim_cfg.sim_dt
        substeps = max(1, int(round(ctrl_dt / sim_dt)))
        capture_every = max(1, int(round(1.0 / (video_fps * sim_dt))))
        video_frames: List[np.ndarray] = []

        # ── Offscreen renderer (shared by both modes) ──────────────────────
        renderer = None
        vcam     = None
        if save_video:
            try:
                renderer = mujoco.Renderer(self.model, height=480, width=640)
                vcam = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(vcam)
                vcam.lookat[:]  = _lookat
                vcam.distance   = viewer_distance
                vcam.elevation  = viewer_elevation
                vcam.azimuth    = viewer_azimuth
                print("[VIDEO] Offscreen renderer ready (640×480).")
            except Exception as exc:
                print(f"[VIDEO] Renderer unavailable: {exc}")
                renderer = None

        sim_time      = 0.0
        phys_step     = 0
        goal_reached  = False
        goal_hold_t   = 0.0          # seconds spent holding at goal (headless only)

        obs_positions: List[np.ndarray] = []
        obs_radii  = np.array([])
        q_dot_opt  = np.zeros(_DOF)
        q_cmd      = self.get_robot_state()[0].copy()

        # ── Shared control-tick logic ──────────────────────────────────────
        def _control_tick():
            nonlocal obs_positions, obs_radii, q_dot_opt, q_cmd, goal_reached, goal_hold_t, target_pos

            q, _      = self.get_robot_state()
            ee_pos, _ = self.compute_forward_kinematics()

            if target_source is not None:
                target_pos = target_source()
                self.set_target_position(target_pos)

            if obstacle_source is not None:
                raw_pos       = obstacle_source()
                obs_positions = raw_pos if raw_pos else []
                obs_radii     = (
                    np.full(len(obs_positions), self.obs_cfg.radius)
                    if obs_positions else np.array([])
                )
                self.update_obstacle_positions(obs_positions)

            if sim_time <= max_duration:
                o_arr = np.array(obs_positions) if obs_positions else np.zeros((0, 3))
                r_arr = obs_radii if len(obs_radii) else np.zeros(0)

                q_dot_opt, info = nmpc_controller.solve(
                    q_init        = q,
                    target_pos    = target_pos,
                    obs_positions = o_arr,
                    obs_radii     = r_arr,
                )
                q_cmd = q + q_dot_opt * ctrl_dt
                self.apply_control(q_cmd)

                if info.get("q_pred") is not None:
                    self.update_trajectory_vis(info["q_pred"])

                obs_pairs = list(zip(
                    obs_positions,
                    obs_radii.tolist() if len(obs_radii) else [],
                ))
                clearance = (
                    min_clearance_numpy(q, obs_pairs)
                    if obs_pairs else float("inf")
                )
                evaluator.record(
                    sim_time      = sim_time,
                    ee_pos        = ee_pos,
                    target_pos    = target_pos,
                    q_dot         = q_dot_opt,
                    clearance     = clearance,
                    solve_time    = info["solve_time"],
                    solver_status = info["status"],
                )

                dist = float(np.linalg.norm(ee_pos - target_pos))
                if not goal_reached and dist < goal_tol:
                    goal_reached = True
                    print(
                        f"\n[SIM] Goal reached!  "
                        f"t={sim_time:.2f}s  dist={dist*100:.1f} cm  "
                        f"clearance={clearance*100:.1f} cm"
                    )
                if goal_reached:
                    goal_hold_t += ctrl_dt
            else:
                self.apply_control(q_cmd)

        # ── Shared frame-capture + physics step ───────────────────────────
        def _step_and_capture():
            nonlocal phys_step
            mujoco.mj_step(self.model, self.data)
            phys_step += 1
            if (renderer is not None
                    and phys_step % capture_every == 0
                    and len(video_frames) < max_video_frames):
                renderer.update_scene(self.data, camera=vcam)
                video_frames.append(renderer.render().copy())

        # ── HEADLESS branch ────────────────────────────────────────────────
        if headless:
            print(f"[SIM] Headless run — max {max_duration:.0f} s.  "
                  f"Target = {np.round(target_pos, 3)}")
            while sim_time <= max_duration:
                if phys_step % substeps == 0:
                    _control_tick()
                    sim_time += ctrl_dt
                    # Stop 2 s after goal is held (give time for video capture)
                    if goal_hold_t >= 2.0:
                        break
                _step_and_capture()

        # ── VIEWER branch ──────────────────────────────────────────────────
        else:
            print("[SIM] Viewer open — close the window or press Ctrl-C to exit.")
            print(f"[SIM] Controller @ {self.sim_cfg.control_freq} Hz.  "
                  f"Target = {np.round(target_pos, 3)}")

            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.lookat[:]  = _lookat
                viewer.cam.distance   = viewer_distance
                viewer.cam.elevation  = viewer_elevation
                viewer.cam.azimuth    = viewer_azimuth

                while viewer.is_running():
                    step_wall_start = time.perf_counter()

                    if phys_step % substeps == 0:
                        _control_tick()
                        sim_time += ctrl_dt
                        if sim_time > max_duration and not goal_reached:
                            print(
                                f"[SIM] Max duration ({max_duration:.0f}s) reached. "
                                "Close viewer window to save & exit."
                            )

                    _step_and_capture()
                    viewer.sync()

                    elapsed = time.perf_counter() - step_wall_start
                    sleep_t = sim_dt - elapsed
                    if sleep_t > 0.0001:
                        time.sleep(sleep_t)

        if renderer is not None:
            renderer.close()

        print(f"[SIM] Session ended. Captured {len(video_frames)} video frames.")
        return evaluator.summary(), video_frames
