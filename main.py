"""
main.py
-------
Entry point for Vision-Based Dynamic Obstacle Avoidance for UR5e.

Usage:
    python main.py                          # NMPC + simulated obstacles
    python main.py --mode camera            # NMPC + live webcam (requires cv2 + ultralytics)
    python main.py --mode compare           # compare NMPC vs APF vs RRTConnect
    python main.py --target 0.5 0.2 0.5    # set custom target end-effector position

Pipeline:
    Webcam → YOLO → ArUco → obstacle_positions (world frame)
    → KinematicNMPC → q_dot → MuJoCo simulation
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

from config import (
    NMPCConfig, ObstacleConfig, SimConfig, BaselineConfig,
    UR5E_HOME_Q,
)
from ur5e_kinematics import fk_numpy, min_clearance_numpy
from nmpc_controller import KinematicNMPC
from mujoco_sim import UR5eSimulation
from evaluation import EpisodeEvaluator, compare_methods
from baselines import APFController, RRTConnectPlanner, run_apf_episode


# ---------------------------------------------------------------------------
# Fixed obstacle source  (default demo mode)
# ---------------------------------------------------------------------------

# One obstacle placed directly on the straight-line path between the arm's
# home end-effector position (-0.13, 0.49, 0.49) and the default target
# (0.40, 0.00, 0.40).  The NMPC must route the arm around it.
#
#   Obstacle 0  t≈0.40 along path → (0.09, 0.29, 0.44)

FIXED_OBSTACLE_POSITIONS: List[np.ndarray] = [
    np.array([0.09,  0.29, 0.44]),
]


class FixedObstacleSource:
    """
    Returns static obstacle positions every call.
    The positions are hard-coded to block the straight-line path from the
    home configuration end-effector to the default target, forcing a
    visible avoidance manoeuvre.
    """

    def __call__(self) -> List[np.ndarray]:
        return [p.copy() for p in FIXED_OBSTACLE_POSITIONS]


# ---------------------------------------------------------------------------
# Moving obstacle source  (optional, shows dynamic re-planning)
# ---------------------------------------------------------------------------

class MovingObstacleSource:
    """
    Slowly oscillates the first obstacle left-right while the second stays fixed.
    Use --mode moving to activate.
    """

    def __init__(self):
        self._t = 0.0

    def __call__(self) -> List[np.ndarray]:
        self._t += 0.20            # incremented every NMPC step (~5 Hz)
        p0 = FIXED_OBSTACLE_POSITIONS[0].copy()
        p0[0] += 0.08 * np.sin(self._t * 0.5)   # oscillate ±8 cm in X
        return [p0]


class SweepingTargetSource:
    """
    Returns a target EE position that sweeps left-right in the Y axis (sine wave).
    Used by cam_sweep mode so the robot keeps moving while NMPC avoids obstacles.
    """

    def __init__(
        self,
        center_pos:  np.ndarray = None,
        amplitude_y: float = 0.30,
        period_s:    float = 12.0,
    ):
        self._center = (center_pos.copy() if center_pos is not None
                        else np.array([0.40, 0.0, 0.40]))
        self._amp_y  = amplitude_y
        self._period = period_s
        self._t0     = time.perf_counter()

    def __call__(self) -> np.ndarray:
        t = time.perf_counter() - self._t0
        # Triangle wave: constant sweep speed on both ขาไป and ขากลับ
        # phase [0, 0.5) → Y goes from -amp to +amp  (ขาไป)
        # phase [0.5, 1)  → Y goes from +amp to -amp  (ขากลับ)
        phase = (t % self._period) / self._period
        if phase < 0.5:
            frac = phase * 2.0          # 0 → 1
        else:
            frac = (1.0 - phase) * 2.0  # 1 → 0
        y = self._amp_y * (2.0 * frac - 1.0)   # -amp → +amp → -amp
        pos = self._center.copy()
        pos[1] = self._center[1] + y
        return pos


# ---------------------------------------------------------------------------
# Camera obstacle source (live)
# ---------------------------------------------------------------------------

def _try_import_vision():
    try:
        from obstacle_localization import VisionObstacleTracker
        from bottle_detector import DetectorConfig
        return VisionObstacleTracker, DetectorConfig
    except ImportError as e:
        print(f"[WARNING] Vision pipeline unavailable: {e}")
        return None, None


class CameraNoArucoObstacleSource:
    """
    YOLO-only obstacle source — no ArUco needed.
    Estimates 3-D bottle positions from bounding-box size (rough depth).

    Overlay shows:
      - Estimated world [x, y, z] coordinates below each detected bounding box
      - Bottom info panel: sweep workspace Y-bar, current target, obstacle list
    """

    def __init__(
        self,
        show_feed:       bool        = True,
        camera_index:    int         = 0,
        target_source                = None,   # SweepingTargetSource (for display)
        sweep_center:    np.ndarray  = None,   # [x, y, z] sweep center (m)
        sweep_amplitude_y: float     = 0.30,   # amplitude in Y (m)
    ):
        self._cap        = None
        self._model_yolo = None
        self._show       = show_feed
        self._fps        = 0.0
        self._prev_t     = time.perf_counter()
        self._frame_n    = 0
        self._target_src   = target_source
        self._sweep_center = (sweep_center.copy() if sweep_center is not None
                              else np.array([0.40, 0.0, 0.40]))
        self._sweep_amp_y  = sweep_amplitude_y
        try:
            import cv2 as _cv2
            from bottle_detector import (
                DetectorConfig, load_yolo,
                parse_bottle_detections, draw_bottle_detections,
            )
            cfg = DetectorConfig()
            print("[CAM_SWEEP] Loading YOLO model …")
            self._model_yolo, self._bottle_class_id = load_yolo(cfg)
            print(f"[CAM_SWEEP] Opening camera index {camera_index} …")
            cap = _cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera at index {camera_index}")
            # MJPG + 640x480: avoids black-frame issue on cameras that
            # don't support the 1280x720 default in DetectorConfig
            cap.set(_cv2.CAP_PROP_FOURCC, _cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(_cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                cap.release()
                raise RuntimeError(f"Camera {camera_index} opened but returned no frame")
            actual_h, actual_w = test_frame.shape[:2]
            print(f"[CAM_SWEEP] Camera ready — {actual_w}x{actual_h} px.")
            self._cap   = cap
            self._cfg   = cfg
            self._cv2   = _cv2
            self._parse = parse_bottle_detections
            self._draw  = draw_bottle_detections
            self._font  = _cv2.FONT_HERSHEY_SIMPLEX
        except Exception as exc:
            print(f"[WARNING] CameraNoArucoObstacleSource unavailable: {exc}")

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def _draw_obstacle_coords(self, frame, bottles, world_pos):
        """Draw estimated world [x,y,z] below each bounding box."""
        cv2 = self._cv2
        for bot, wp in zip(bottles, world_pos):
            x1, _y1, _x2, y2 = bot.bbox_xyxy
            txt = f"[{wp[0]:+.2f},{wp[1]:+.2f},{wp[2]:+.2f}]m"
            cv2.putText(frame, txt, (int(x1), int(y2) + 18),
                        self._font, 0.48, (0, 255, 120), 2, cv2.LINE_AA)

    def _draw_info_panel(self, frame, world_pos):
        """Dark bottom panel: sweep Y-bar + current target + obstacle list."""
        cv2   = self._cv2
        h, w  = frame.shape[:2]
        panel_h = 110
        py0     = h - panel_h

        # Dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, py0), (w, h), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.line(frame, (0, py0), (w, py0), (80, 80, 80), 1)

        y = py0 + 18

        # ── Sweep Y workspace bar ──────────────────────────────────────
        bar_x0  = 10
        bar_x1  = w - 10
        bar_y   = y
        bar_w   = bar_x1 - bar_x0
        ctr_y   = self._sweep_center[1]
        amp     = self._sweep_amp_y
        y_min   = ctr_y - amp
        y_max   = ctr_y + amp

        cv2.line(frame, (bar_x0, bar_y), (bar_x1, bar_y), (100, 100, 100), 2)
        tick_h = 5
        for tick_y in [y_min, ctr_y, y_max]:
            tx = int(bar_x0 + (tick_y - y_min) / (y_max - y_min + 1e-9) * bar_w)
            cv2.line(frame, (tx, bar_y - tick_h), (tx, bar_y + tick_h), (160, 160, 160), 2)
        cv2.putText(frame, f"{y_min:+.2f}", (bar_x0, bar_y - 7),
                    self._font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{y_max:+.2f}", (bar_x1 - 32, bar_y - 7),
                    self._font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Y: [{y_min:+.2f} → {y_max:+.2f}] m",
                    (bar_x0 + bar_w // 2 - 60, bar_y - 7),
                    self._font, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        # Current target marker (cyan triangle)
        if self._target_src is not None:
            tgt = self._target_src()
            tgt_y_clamp = float(np.clip(tgt[1], y_min, y_max))
            tx = int(bar_x0 + (tgt_y_clamp - y_min) / (y_max - y_min + 1e-9) * bar_w)
            pts = np.array([[tx, bar_y + 4], [tx - 6, bar_y + 14], [tx + 6, bar_y + 14]])
            cv2.fillPoly(frame, [pts], (0, 200, 255))
        else:
            tgt = None

        y += 26

        # ── Target position text ───────────────────────────────────────
        if tgt is not None:
            cv2.putText(frame,
                        f"Target:  [{tgt[0]:+.3f}, {tgt[1]:+.3f}, {tgt[2]:+.3f}] m",
                        (10, y), self._font, 0.52, (0, 200, 255), 2, cv2.LINE_AA)
            y += 20

        # ── Obstacle coordinates ───────────────────────────────────────
        if world_pos:
            obs_txts = [f"Obs{i}:[{wp[0]:+.3f},{wp[1]:+.3f},{wp[2]:+.3f}]m"
                        for i, wp in enumerate(world_pos)]
            cv2.putText(frame, "  ".join(obs_txts),
                        (10, y), self._font, 0.48, (0, 255, 120), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No obstacles detected",
                        (10, y), self._font, 0.48, (100, 100, 100), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------

    def __call__(self) -> List[np.ndarray]:
        if self._cap is None or self._model_yolo is None:
            return []
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return []

        yolo_res = self._model_yolo.predict(
            frame,
            conf=self._cfg.conf_threshold,
            iou=self._cfg.iou_threshold,
            verbose=False,
            device="cpu",
        )
        bottles   = self._parse(
            yolo_res[0], self._bottle_class_id, self._cfg.conf_threshold
        )
        # Place obstacle at the MID-POINT of the sweep (not at the extreme).
        # sweep range = [center_y - amp, center_y + amp]
        # mid-right = center_y + amp*0.5  → robot crosses it going AND coming back
        # (placing at +amp would only be crossed going right, not returning left)
        _workspace_y = float(self._sweep_center[1] + self._sweep_amp_y * 0.5)
        world_pos = _estimate_positions_no_aruco(
            bottles, frame.shape,
            fixed_y = _workspace_y,
            z_fixed = float(self._sweep_center[2]),
        )

        # Terminal printout every 30 frames
        self._frame_n += 1
        if self._frame_n % 30 == 0 and world_pos:
            for i, wp in enumerate(world_pos):
                print(f"[CAM_SWEEP] obstacle_{i}: "
                      f"[{wp[0]:+.3f}, {wp[1]:+.3f}, {wp[2]:+.3f}] m")

        if self._show:
            self._draw(frame, bottles)
            self._draw_obstacle_coords(frame, bottles, world_pos)

            now = time.perf_counter()
            self._fps = 0.9 * self._fps + 0.1 / max(now - self._prev_t, 1e-6)
            self._prev_t = now
            self._cv2.putText(
                frame,
                f"FPS:{self._fps:.1f}  Bottles:{len(bottles)}  Obstacles:{len(world_pos)}",
                (10, 30), self._font, 0.7, (0, 255, 255), 2, self._cv2.LINE_AA,
            )
            self._cv2.putText(
                frame, "No ArUco — rough depth estimate",
                (10, 60), self._font, 0.6, (0, 165, 255), 2, self._cv2.LINE_AA,
            )
            self._draw_info_panel(frame, world_pos)
            self._cv2.imshow("CAM_SWEEP — Bottle Detection  [q = quit MuJoCo]", frame)
            self._cv2.waitKey(1)

        return world_pos

    def release(self):
        if self._cap is not None:
            self._cap.release()
        if self._show:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass


class CameraObstacleSource:
    """
    Reads from webcam, runs YOLO+ArUco, returns world obstacle positions.
    Falls back to empty list if camera unavailable.
    """
    def __init__(self):
        VisionObstacleTracker, DetectorConfig = _try_import_vision()
        if VisionObstacleTracker is None:
            self._tracker = None
            self._cap     = None
            print("[WARNING] Running without live camera.")
            return

        cfg            = DetectorConfig()
        self._tracker  = VisionObstacleTracker(cfg)
        try:
            import cv2
            self._cap = self._tracker.open_camera()
            self._cv2 = cv2
        except Exception as e:
            print(f"[WARNING] Camera open failed: {e}")
            self._cap = None

    def __call__(self) -> List[np.ndarray]:
        if self._tracker is None or self._cap is None:
            return []
        ret, frame = self._cap.read()
        if not ret:
            return []
        return self._tracker.update(frame)

    def release(self):
        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
# NMPC mode
# ---------------------------------------------------------------------------

def run_nmpc(
    target_pos:    np.ndarray,
    obstacle_mode: str,
    max_duration:  float = 60.0,
    n_obs_max:     int   = 5,
) -> dict:
    nmpc_cfg = NMPCConfig()
    sim_cfg  = SimConfig()

    # Sphere mode uses a larger radius (0.10 m) so the visual sphere and the
    # NMPC collision model are in agreement, and avoidance is more dramatic.
    if obstacle_mode == "sphere":
        obs_cfg = ObstacleConfig(radius=0.10)
        sim_cfg.scene_xml = "scene_sphere_obstacle.xml"
    else:
        obs_cfg = ObstacleConfig()

    print("\n" + "=" * 60)
    print("  UR5e Kinematic NMPC — Obstacle Avoidance Demo")
    print("=" * 60)
    print(f"  Target         : {np.round(target_pos, 3)}")
    print(f"  Obstacle 0     : {FIXED_OBSTACLE_POSITIONS[0]} (on path)")
    print(f"  Obs shape      : {'sphere' if obstacle_mode == 'sphere' else 'cylinder'}")
    print(f"  Obs radius     : {obs_cfg.radius} m")
    print(f"  Safety margin  : {obs_cfg.safety_margin} m  "
          f"(exclusion zone = {obs_cfg.radius + obs_cfg.safety_margin:.2f} m)")
    print(f"  Mode           : {obstacle_mode}")
    print("=" * 60)

    print("\n[INIT] Building NMPC solver (~5 s) …")
    t0 = time.perf_counter()
    controller = KinematicNMPC(
        cfg     = nmpc_cfg,
        obs_cfg = obs_cfg,
        n_obs   = n_obs_max,
    )
    print(f"[INIT] Solver ready in {time.perf_counter()-t0:.1f} s")
    print("[INIT] Viewer opening… first NMPC solve takes ~3-5 s, then robot moves.\n")

    sim = UR5eSimulation(sim_cfg=sim_cfg, nmpc_cfg=nmpc_cfg, obs_cfg=obs_cfg)

    # Choose obstacle source
    if obstacle_mode == "camera":
        cam_src    = CameraObstacleSource()
        obs_source = cam_src
    elif obstacle_mode == "moving":
        obs_source = MovingObstacleSource()
    else:
        # sim / sphere: single fixed obstacle blocking the straight-line path
        obs_source = FixedObstacleSource()

    results, video_frames = sim.run(
        nmpc_controller = controller,
        target_pos      = target_pos,
        obstacle_source = obs_source,
        max_duration    = max_duration,
        save_video      = True,
        video_fps       = 10,
    )

    if obstacle_mode == "camera":
        cam_src.release()

    # ── Save video + plots ──────────────────────────────────────────────────
    print("\n[POST] Saving video and trajectory plots…")
    from plot_results import save_all
    active_obs = (
        obs_source._positions if hasattr(obs_source, "_positions")
        else FIXED_OBSTACLE_POSITIONS
    )
    paths = save_all(
        results            = results,
        video_frames       = video_frames,
        obstacle_positions = active_obs,
        obstacle_radius    = obs_cfg.radius,
        safety_margin      = obs_cfg.safety_margin,
        fps                = 10,
        tag                = obstacle_mode,
    )
    print("\n[POST] Artifacts saved:")
    for name, path in paths.items():
        print(f"  {name:<16s} → {path}")

    return results


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def run_comparison(target_pos: np.ndarray) -> None:
    """
    Run NMPC, APF, and RRTConnect in MuJoCo — each headlessly — and save
    a separate video for each algorithm plus a side-by-side metrics table.
    """
    from baselines import APFControllerMuJoCo, RRTPathExecutor
    from plot_results import save_video as _save_video, save_all

    obs_cfg  = ObstacleConfig()
    bl_cfg   = BaselineConfig()
    nmpc_cfg = NMPCConfig()
    sim_cfg  = SimConfig()
    ctrl_dt  = 1.0 / sim_cfg.control_freq

    obs_pairs = [(p.copy(), obs_cfg.radius) for p in FIXED_OBSTACLE_POSITIONS]

    all_results: dict = {}
    all_videos:  dict = {}

    obs_source = FixedObstacleSource()
    sim = UR5eSimulation(sim_cfg=sim_cfg, nmpc_cfg=nmpc_cfg, obs_cfg=obs_cfg)

    # ── 1. NMPC ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[COMPARE 1/3] NMPC — building solver …")
    nmpc_ctrl = KinematicNMPC(nmpc_cfg, obs_cfg, n_obs=len(obs_pairs))
    print("[COMPARE 1/3] Running NMPC in headless MuJoCo …")
    results_nmpc, frames_nmpc = sim.run(
        nmpc_controller = nmpc_ctrl,
        target_pos      = target_pos,
        obstacle_source = obs_source,
        max_duration    = 60.0,
        save_video      = True,
        headless        = True,
    )
    all_results["NMPC"] = results_nmpc
    all_videos["NMPC"]  = frames_nmpc
    print(f"[COMPARE 1/3] NMPC done — success={results_nmpc.get('success')}, "
          f"path_length={results_nmpc.get('path_length_m', float('nan')):.3f} m")

    # ── 2. APF ───────────────────────────────────────────────────────────────
    print("\n[COMPARE 2/3] Running APF in headless MuJoCo …")
    apf_ctrl = APFControllerMuJoCo(bl_cfg, obs_cfg)
    results_apf, frames_apf = sim.run(
        nmpc_controller = apf_ctrl,
        target_pos      = target_pos,
        obstacle_source = obs_source,
        max_duration    = 60.0,
        save_video      = True,
        headless        = True,
    )
    all_results["APF"] = results_apf
    all_videos["APF"]  = frames_apf
    print(f"[COMPARE 2/3] APF done — success={results_apf.get('success')}, "
          f"path_length={results_apf.get('path_length_m', float('nan')):.3f} m")

    # ── 3. RRTConnect ────────────────────────────────────────────────────────
    print("\n[COMPARE 3/3] RRTConnect — planning …")
    rrt = RRTConnectPlanner(bl_cfg, obs_cfg)
    q_goal = _ik_approx(target_pos, UR5E_HOME_Q.copy(), max_iter=500)
    print(f"[COMPARE 3/3] IK goal found — EE error "
          f"{np.linalg.norm(fk_numpy(q_goal)[0] - target_pos)*100:.1f} cm")

    path, rrt_solve_time = rrt.plan(UR5E_HOME_Q.copy(), q_goal, obs_pairs)

    if path:
        print(f"[COMPARE 3/3] RRTConnect found path ({len(path)} nodes) "
              f"in {rrt_solve_time:.2f} s — executing in MuJoCo …")
        rrt_exec = RRTPathExecutor(path, ctrl_dt=ctrl_dt)
        rrt_max_dur = len(path) * ctrl_dt * 2.0 + 5.0
        results_rrt, frames_rrt = sim.run(
            nmpc_controller = rrt_exec,
            target_pos      = target_pos,
            obstacle_source = obs_source,
            max_duration    = rrt_max_dur,
            save_video      = True,
            headless        = True,
        )
    else:
        print(f"[COMPARE 3/3] RRTConnect FAILED — no path found in "
              f"{rrt_solve_time:.2f} s ({bl_cfg.rrt_max_iter} iters). "
              "Increase rrt_max_iter in config.py if this persists.")
        results_rrt = {
            "success":               False,
            "path_length_m":         float("nan"),
            "min_clearance_m":       float("nan"),
            "mean_clearance_m":      float("nan"),
            "tracking_rms_m":        float("nan"),
            "tracking_mean_m":       float("nan"),
            "end_effector_error_m":  float("nan"),
            "mean_solve_time_ms":    rrt_solve_time * 1e3,
            "max_solve_time_ms":     rrt_solve_time * 1e3,
            "constraint_violations": 0,
            "n_steps":               0,
            "trajectory":            {},
        }
        frames_rrt = []

    all_results["RRTConnect"] = results_rrt
    all_videos["RRTConnect"]  = frames_rrt
    print(f"[COMPARE 3/3] RRTConnect done — success={results_rrt.get('success')}")

    # ── Save per-algorithm videos ─────────────────────────────────────────────
    import os
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n[COMPARE] Saving videos …")
    for method, frames in all_videos.items():
        if frames:
            path_out = os.path.join(results_dir, f"compare_{method}.gif")
            _save_video(frames, filename=path_out, fps=10)

    # Save NMPC trajectory plot as well
    if all_results["NMPC"].get("trajectory"):
        save_all(
            results            = all_results["NMPC"],
            video_frames       = [],            # already saved above
            obstacle_positions = FIXED_OBSTACLE_POSITIONS,
            obstacle_radius    = obs_cfg.radius,
            safety_margin      = obs_cfg.safety_margin,
            tag                = "compare_NMPC",
        )

    # ── Comparison table ───────────────────────────────────────────────────────
    compare_methods(all_results)


def _ik_approx(
    target_pos: np.ndarray,
    q_init:     np.ndarray,
    max_iter:   int = 300,
    lr:         float = 0.5,
    tol:        float = 1e-3,
) -> np.ndarray:
    """Simple gradient-descent IK approximation."""
    from ur5e_kinematics import jacobian_numpy
    from config import UR5E_Q_MIN, UR5E_Q_MAX
    q = q_init.copy()
    for _ in range(max_iter):
        ee, _ = fk_numpy(q)
        err = target_pos - ee
        if np.linalg.norm(err) < tol:
            break
        J     = jacobian_numpy(q)
        J_pinv = np.linalg.pinv(J)
        dq    = lr * J_pinv @ err
        q     = np.clip(q + dq, UR5E_Q_MIN, UR5E_Q_MAX)
    return q


def _path_ee_length(path: list) -> float:
    total = 0.0
    prev, _ = fk_numpy(path[0])
    for qi in path[1:]:
        cur, _ = fk_numpy(qi)
        total += np.linalg.norm(cur - prev)
        prev = cur
    return total


def _path_min_clearance(
    path:      list,
    obstacles: list,
) -> float:
    min_d = float("inf")
    for q in path:
        d = min_clearance_numpy(q, obstacles)
        if d < min_d:
            min_d = d
    return min_d


# ---------------------------------------------------------------------------
# Cam-sweep NMPC mode  (camera obstacle detection + sweeping target + NMPC)
# ---------------------------------------------------------------------------

def run_cam_sweep_nmpc(
    center_pos:   np.ndarray = None,
    amplitude_y:  float = 0.30,
    period_s:     float = 12.0,
    max_duration: float = 300.0,
    camera_index: int   = 0,
) -> None:
    """
    cam_sweep mode: UR5e sweeps left-right in Y axis while NMPC avoids
    bottle obstacles detected by YOLO (no ArUco required).

    The target oscillates as:  target_y = center_y + amplitude_y * sin(2π t / period_s)

    Use this to stress-test NMPC with real camera input before adding ArUco.
    Close the MuJoCo viewer window to quit.
    """
    if center_pos is None:
        center_pos = np.array([0.40, 0.0, 0.40])

    nmpc_cfg = NMPCConfig()
    sim_cfg  = SimConfig()
    obs_cfg  = ObstacleConfig()

    print("\n" + "=" * 60)
    print("  UR5e Cam-Sweep NMPC Mode")
    print("=" * 60)
    print(f"  Sweep center   : {np.round(center_pos, 3)}")
    print(f"  Amplitude Y    : ±{amplitude_y:.2f} m  "
          f"(Y range [{center_pos[1]-amplitude_y:.2f}, {center_pos[1]+amplitude_y:.2f}])")
    print(f"  Period         : {period_s:.1f} s  (left → right → left)")
    print(f"  Obstacle radius: {obs_cfg.radius} m  +  {obs_cfg.safety_margin} m margin")
    print(f"  ArUco          : NOT required (rough pinhole depth estimate)")
    print(f"  Max duration   : {max_duration:.0f} s  (close viewer to quit early)")
    print("=" * 60)

    print("\n[INIT] Building NMPC solver (~5 s) …")
    t0 = time.perf_counter()
    controller = KinematicNMPC(cfg=nmpc_cfg, obs_cfg=obs_cfg, n_obs=5)
    print(f"[INIT] Solver ready in {time.perf_counter()-t0:.1f} s")

    target_src = SweepingTargetSource(center_pos, amplitude_y, period_s)
    cam_src    = CameraNoArucoObstacleSource(
        show_feed         = True,
        camera_index      = camera_index,
        target_source     = target_src,
        sweep_center      = center_pos,
        sweep_amplitude_y = amplitude_y,
    )

    sim = UR5eSimulation(sim_cfg=sim_cfg, nmpc_cfg=nmpc_cfg, obs_cfg=obs_cfg)

    print("[INIT] Viewer opening — camera window will appear after first frame.\n")

    sim.run(
        nmpc_controller = controller,
        target_pos      = center_pos.copy(),
        obstacle_source = cam_src,
        target_source   = target_src,
        max_duration    = max_duration,
        save_video      = False,
    )

    cam_src.release()
    print("\n[CAM_SWEEP] Done.")


# ---------------------------------------------------------------------------
# CV test mode  (vision pipeline + MuJoCo viewer, no NMPC)
# ---------------------------------------------------------------------------

# Bottle height in metres — used for depth estimation without ArUco
_BOTTLE_REAL_HEIGHT_M = 0.23


def _estimate_positions_no_aruco(
    bottles,
    frame_shape:     Tuple,
    fixed_y:         float = None,   # if set: skip depth estimate, use this world Y [m]
    assumed_depth_m: float = 0.50,   # assumed camera depth used only for X when fixed_y set
    cam_dist_m:      float = 0.80,   # camera distance from robot base, used when fixed_y=None
    y_min:           float = 0.05,
    y_max:           float = 0.65,
    z_fixed:         float = 0.40,
) -> List[np.ndarray]:
    """
    Rough 3-D estimate from bounding-box size alone (no ArUco).

    Two modes:
      fixed_y=None  — depth from bounding-box height (unreliable, kept for cv_test/cam_test)
      fixed_y=float — obstacle placed at that Y; only X is computed from camera
                      (recommended for cam_sweep where lateral placement matters more than depth)

    CAMERA_MATRIX in config.py is designed for 1280×720; this function scales
    the intrinsics to the actual captured frame size automatically.
    """
    from config import CAMERA_MATRIX
    h_frame, w_frame = frame_shape[:2]
    sx    = w_frame / (CAMERA_MATRIX[0, 2] * 2)
    sy    = h_frame / (CAMERA_MATRIX[1, 2] * 2)
    fx    = float(CAMERA_MATRIX[0, 0]) * sx
    fy    = float(CAMERA_MATRIX[1, 1]) * sy
    cx_px = float(CAMERA_MATRIX[0, 2]) * sx

    positions: List[np.ndarray] = []
    for b in bottles:
        h_px = int(b.bbox_xyxy[3]) - int(b.bbox_xyxy[1])
        if h_px < 10:
            continue

        if fixed_y is not None:
            # X from camera using an assumed depth; Y fixed inside workspace
            x_cam = (b.center_x - cx_px) * assumed_depth_m / fx
            x_w   = float(np.clip(x_cam, -0.55, 0.55))
            y_w   = float(fixed_y)
        else:
            # Depth from apparent bottle height
            z_cam = _BOTTLE_REAL_HEIGHT_M * fy / h_px
            x_cam = (b.center_x - cx_px) * z_cam / fx
            x_w   = float(np.clip(x_cam, -0.7, 0.7))
            y_w   = float(np.clip(cam_dist_m - z_cam, y_min, y_max))

        positions.append(np.array([x_w, y_w, z_fixed]))
    return positions


def run_cv_test(target_pos: np.ndarray) -> None:
    """
    CV test mode: live camera window (YOLO + ArUco overlays) + MuJoCo viewer.
    Robot stays at home pose — no NMPC. Use this to verify the vision pipeline
    before running the full camera mode.

    With ArUco marker:  obstacle position is accurate (world-frame).
    Without ArUco:      rough depth estimate shown with a yellow WARNING overlay.
    Press 'q' in the camera window to quit.
    """
    import cv2 as _cv2
    from bottle_detector import (
        DetectorConfig, build_capture, load_yolo, build_aruco_detector,
        parse_bottle_detections, detect_aruco,
        draw_bottle_detections, draw_aruco_detections, draw_stats,
    )
    from obstacle_localization import bottles_to_world
    import mujoco
    import mujoco.viewer

    cfg = DetectorConfig()
    print("\n[CV_TEST] Loading YOLO model …")
    model, bottle_class_id = load_yolo(cfg)
    aruco_detector = build_aruco_detector(cfg)

    print("[CV_TEST] Opening camera …")
    cap = build_capture(cfg)

    print("[CV_TEST] Loading MuJoCo scene …")
    sim_cfg  = SimConfig()
    obs_cfg  = ObstacleConfig()
    nmpc_cfg = NMPCConfig()
    sim = UR5eSimulation(sim_cfg=sim_cfg, nmpc_cfg=nmpc_cfg, obs_cfg=obs_cfg)

    print("\n[CV_TEST] Running.  Press 'q' in the camera window to quit.")
    print("[CV_TEST] Show a bottle to the camera.")
    print("[CV_TEST]   With ArUco marker  → accurate world-frame position.")
    print("[CV_TEST]   Without ArUco      → rough estimate (yellow warning).\n")

    font      = _cv2.FONT_HERSHEY_SIMPLEX
    fps       = 0.0
    prev_tick = time.perf_counter()
    frame_n   = 0

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        sim.set_target_position(target_pos)

        while viewer.is_running():
            ret, frame = cap.read()
            if not ret:
                continue

            # -- Detection --------------------------------------------------
            yolo_res  = model.predict(
                frame,
                conf=cfg.conf_threshold,
                iou=cfg.iou_threshold,
                verbose=False,
                device="cpu",
            )
            bottles = parse_bottle_detections(
                yolo_res[0], bottle_class_id, cfg.conf_threshold
            )
            markers = detect_aruco(frame, aruco_detector)

            # -- World positions --------------------------------------------
            has_aruco   = len(markers) > 0
            has_bottles = len(bottles) > 0

            if has_aruco and has_bottles:
                world_pos = bottles_to_world(bottles, markers)
                pos_source = "ArUco"
            elif has_bottles:
                world_pos = _estimate_positions_no_aruco(bottles, frame.shape)
                pos_source = "ESTIMATE"
            else:
                world_pos = []
                pos_source = "none"

            # -- MuJoCo update ---------------------------------------------
            sim.update_obstacle_positions(world_pos)
            viewer.sync()

            # -- Terminal debug (every 20 frames) --------------------------
            frame_n += 1
            if frame_n % 20 == 0:
                b_str = f"{len(bottles)} bottle(s)" if has_bottles else "no bottles"
                m_str = f"{len(markers)} ArUco"     if has_aruco   else "NO ArUco"
                p_str = (
                    f"{len(world_pos)} obstacle(s) → MuJoCo [{pos_source}]"
                    if world_pos else "no obstacles in MuJoCo"
                )
                print(f"[CV_TEST] {b_str} | {m_str} | {p_str}")
                for i, p in enumerate(world_pos):
                    print(f"          obstacle_{i}: "
                          f"[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] m")

            # -- Camera overlay --------------------------------------------
            draw_bottle_detections(frame, bottles)
            draw_aruco_detections(frame, markers)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - prev_tick, 1e-6)
            prev_tick = now
            draw_stats(frame, fps, len(bottles), len(markers))

            # ArUco status banner
            if has_aruco:
                aruco_txt   = f"ArUco OK ({len(markers)})  — world frame active"
                aruco_color = (0, 220, 0)
            else:
                aruco_txt   = "ArUco NOT found — using rough depth estimate"
                aruco_color = (0, 165, 255)   # orange
            _cv2.putText(frame, aruco_txt, (10, 60), font, 0.6, aruco_color, 2)

            # Obstacle count banner
            if world_pos:
                obs_txt = (f"MuJoCo obstacles: {len(world_pos)}  "
                           f"[{pos_source}]")
                obs_col = (0, 255, 255) if has_aruco else (0, 165, 255)
                _cv2.putText(frame, obs_txt, (10, 90), font, 0.6, obs_col, 2)
            else:
                _cv2.putText(frame, "No obstacles sent to MuJoCo",
                             (10, 90), font, 0.6, (100, 100, 100), 2)

            _cv2.imshow("CV Test — Bottle+ArUco [q=quit]", frame)
            if _cv2.waitKey(1) & 0xFF == ord("q"):
                print("[CV_TEST] 'q' pressed — exiting.")
                break

    cap.release()
    _cv2.destroyAllWindows()
    print("[CV_TEST] Done.")


# ---------------------------------------------------------------------------
# Camera test mode  (no robot, no ArUco — camera feed + MuJoCo bottle markers)
# ---------------------------------------------------------------------------

_CAM_TEST_SCENE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mujoco_menagerie main universal_robots_ur5e",
    "scene_cam_test.xml",
)
_CAM_TEST_MAX_BOTTLES = 5    # must match scene_cam_test.xml
_CAM_TEST_DURATION    = 120  # total run time in seconds (2 minutes)


def _find_camera() -> int:
    """Return the first working camera index (0-4), or -1 if none found."""
    import cv2 as _cv2
    for idx in range(5):
        cap = _cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.release()
            return idx
        cap.release()
    return -1


def _cam_test_update_mujoco(
    model,
    data,
    positions: List[np.ndarray],
) -> None:
    """Move mocap bottle bodies to detected positions; park unused ones far away."""
    for i in range(_CAM_TEST_MAX_BOTTLES):
        body_name = f"bottle_{i}"
        try:
            bid = model.body(body_name).mocapid
        except Exception:
            continue
        if i < len(positions):
            data.mocap_pos[bid] = positions[i]
        else:
            data.mocap_pos[bid] = np.array([10.0, float(i), 0.0])


def run_cam_test(camera_index = -1) -> None:
    """
    Camera test mode: live webcam + YOLO bottle detection displayed in a
    MuJoCo scene (no UR5e robot, no ArUco).

    A 2-second countdown is shown on the camera feed so you have time to
    position the bottle before detection results appear in MuJoCo.
    Press 'q' or close the camera window to quit.
    """
    import cv2 as _cv2
    import mujoco
    import mujoco.viewer
    from bottle_detector import (
        DetectorConfig, load_yolo,
        parse_bottle_detections, draw_bottle_detections,
    )

    # ── Camera setup ─────────────────────────────────────────────────────────
    if isinstance(camera_index, int) and camera_index < 0:
        print("[CAM_TEST] Scanning for camera (indices 0-4) …")
        camera_index = _find_camera()
        if camera_index < 0:
            print(
                "\n[CAM_TEST] ERROR: No camera found.\n"
                "\n"
                "You are running inside WSL2 — USB cameras are not forwarded\n"
                "automatically.  To connect your webcam:\n"
                "\n"
                "  1. On Windows (Admin terminal):\n"
                "       winget install usbipd          # one-time install\n"
                "       usbipd list                    # find your camera BUSID\n"
                "       usbipd bind   --busid <BUSID>  # one-time per camera\n"
                "       usbipd attach --wsl --busid <BUSID>  # every restart\n"
                "\n"
                "  2. In WSL2, verify:  ls /dev/video*\n"
                "\n"
                "  3. Re-run:  python main.py --mode cam_test\n"
                "\n"
                "Or use a video file:\n"
                "       python main.py --mode cam_test --camera /path/to/video.mp4\n"
            )
            return
        print(f"[CAM_TEST] Found camera at index {camera_index}.")

    cfg = DetectorConfig()

    print("\n[CAM_TEST] Loading YOLO model …")
    model_yolo, bottle_class_id = load_yolo(cfg)

    print(f"[CAM_TEST] Opening camera source: {camera_index!r} …")
    cap = _cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[CAM_TEST] ERROR: Cannot open source {camera_index!r}.")
        return
    if isinstance(camera_index, int):
        cap.set(_cv2.CAP_PROP_FOURCC, _cv2.VideoWriter_fourcc(*"MJPG"))

    # Warm-up YOLO
    dummy = np.zeros((cfg.frame_height, cfg.frame_width, 3), dtype=np.uint8)
    model_yolo.predict(dummy, conf=cfg.conf_threshold, iou=cfg.iou_threshold,
                       verbose=False)

    # ── MuJoCo setup (no robot) ───────────────────────────────────────────────
    print(f"[CAM_TEST] Loading MuJoCo scene: {_CAM_TEST_SCENE} …")
    mj_model = mujoco.MjModel.from_xml_path(_CAM_TEST_SCENE)
    mj_data  = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    print(f"\n[CAM_TEST] Ready.  Detection starts now.")
    print("[CAM_TEST] Press 'q' or close the camera window to quit.\n")

    font      = _cv2.FONT_HERSHEY_SIMPLEX
    fps       = 0.0
    prev_tick = time.perf_counter()
    win_name  = "Camera Test — Bottle Detector  [q = quit]"
    start_t   = time.perf_counter()

    print(f"[CAM_TEST] Session will run for {_CAM_TEST_DURATION // 60} minutes.\n")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            elapsed   = time.perf_counter() - start_t
            time_left = max(0.0, _CAM_TEST_DURATION - elapsed)

            # Auto-stop after 2 minutes
            if elapsed >= _CAM_TEST_DURATION:
                print("[CAM_TEST] 2-minute session complete — exiting.")
                break

            # ── Detection starts immediately ──────────────────────────────────
            yolo_res = model_yolo.predict(
                frame,
                conf=cfg.conf_threshold,
                iou=cfg.iou_threshold,
                verbose=False,
                device="cpu",
            )
            bottles   = parse_bottle_detections(
                yolo_res[0], bottle_class_id, cfg.conf_threshold
            )
            world_pos = _estimate_positions_no_aruco(bottles, frame.shape)

            # ── Update MuJoCo ────────────────────────────────────────────────
            _cam_test_update_mujoco(mj_model, mj_data, world_pos)
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # ── Camera overlay ───────────────────────────────────────────────
            draw_bottle_detections(frame, bottles)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - prev_tick, 1e-6)
            prev_tick = now

            # Top bar: FPS + bottle count
            _cv2.putText(
                frame,
                f"FPS:{fps:.1f}  Bottles:{len(bottles)}",
                (10, 30), font, 0.7, (0, 255, 255), 2, _cv2.LINE_AA,
            )

            # Remaining time (top-right)
            m, s = divmod(int(time_left), 60)
            timer_txt = f"{m:01d}:{s:02d}"
            (tw, _), _ = _cv2.getTextSize(timer_txt, font, 0.9, 2)
            _cv2.putText(
                frame, timer_txt,
                (frame.shape[1] - tw - 10, 30),
                font, 0.9, (0, 255, 255), 2, _cv2.LINE_AA,
            )

            _cv2.imshow(win_name, frame)
            if _cv2.waitKey(1) & 0xFF == ord("q"):
                print("[CAM_TEST] 'q' pressed — exiting.")
                break

    cap.release()
    _cv2.destroyAllWindows()
    print("[CAM_TEST] Done.")


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Vision-Based Dynamic Obstacle Avoidance for UR5e"
    )
    p.add_argument(
        "--mode",
        choices=["sim", "sphere", "moving", "camera", "compare", "cv_test", "cam_test", "cam_sweep"],
        default="sim",
        help=(
            "sim       — cylindrical obstacle, NMPC avoidance (default)\n"
            "sphere    — spherical obstacle (r=10 cm), NMPC avoidance\n"
            "moving    — obstacle oscillates, tests dynamic re-planning\n"
            "camera    — live webcam YOLO+ArUco (requires cv2 + ultralytics)\n"
            "compare   — headless benchmark: NMPC vs APF vs RRTConnect (saves 3 videos)\n"
            "cv_test   — test vision pipeline: camera preview + MuJoCo, no NMPC\n"
            "cam_test  — standalone camera test: YOLO bottle detection only, no robot/ArUco\n"
            "cam_sweep — NMPC + camera obstacle detection (no ArUco) + sweeping left-right target"
        ),
    )
    p.add_argument(
        "--amplitude", type=float, default=0.30,
        help="cam_sweep: sweep amplitude in Y axis [m] (default 0.30)",
    )
    p.add_argument(
        "--period", type=float, default=12.0,
        help="cam_sweep: sweep period in seconds (default 12.0)",
    )
    p.add_argument(
        "--target", nargs=3, type=float, metavar=("X", "Y", "Z"),
        default=[0.40, 0.00, 0.40],
        help="Target EE position in world frame [m]  (default: 0.40 0.00 0.40)",
    )
    p.add_argument(
        "--duration", type=float, default=60.0,
        help="Seconds of NMPC control; viewer stays open after (default 60)",
    )
    p.add_argument(
        "--camera", default=None,
        help=(
            "cam_test only: camera index (int) or video file path. "
            "Omit to auto-detect."
        ),
    )
    return p.parse_args()


def main():
    args = _parse_args()
    target = np.array(args.target, dtype=float)

    print(f"\n[MAIN] Mode    : {args.mode}")
    print(f"[MAIN] Target  : {target}")

    if args.mode == "compare":
        run_comparison(target)
    elif args.mode == "cam_sweep":
        cam_idx = 0
        if args.camera is not None:
            try:
                cam_idx = int(args.camera)
            except ValueError:
                cam_idx = args.camera
        run_cam_sweep_nmpc(
            center_pos   = target,
            amplitude_y  = args.amplitude,
            period_s     = args.period,
            max_duration = args.duration,
            camera_index = cam_idx,
        )
    elif args.mode == "cv_test":
        run_cv_test(target)
    elif args.mode == "cam_test":
        cam_arg = args.camera
        if cam_arg is not None:
            try:
                cam_arg = int(cam_arg)
            except ValueError:
                pass  # keep as string (video file path)
        run_cam_test(camera_index=-1 if cam_arg is None else cam_arg)
    elif args.mode == "sphere":
        results = run_nmpc(
            target_pos     = target,
            obstacle_mode  = "sphere",
            max_duration   = args.duration,
        )
        print("\n[MAIN] Episode complete.")
        skip = {"trajectory"}
        for k, v in results.items():
            if k in skip:
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        results = run_nmpc(
            target_pos     = target,
            obstacle_mode  = args.mode,
            max_duration   = args.duration,
        )
        print("\n[MAIN] Episode complete.")
        skip = {"trajectory"}
        for k, v in results.items():
            if k in skip:
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
