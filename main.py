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
# CV test mode  (vision pipeline + MuJoCo viewer, no NMPC)
# ---------------------------------------------------------------------------

# Bottle height in metres — used for depth estimation without ArUco
_BOTTLE_REAL_HEIGHT_M = 0.23


def _estimate_positions_no_aruco(
    bottles,
    frame_shape: Tuple,
) -> List[np.ndarray]:
    """
    Rough 3-D estimate from bounding-box size alone (no ArUco).
    Uses pinhole model + assumed bottle height to get depth, then
    a fixed camera-to-world transform guess (camera ~1 m in front of robot,
    50 cm above table, looking toward origin).
    Positions are approximate — good enough to verify detection is working.
    """
    from config import CAMERA_MATRIX
    fx = float(CAMERA_MATRIX[0, 0])
    fy = float(CAMERA_MATRIX[1, 1])
    cx_px = float(CAMERA_MATRIX[0, 2])
    cy_px = float(CAMERA_MATRIX[1, 2])

    positions: List[np.ndarray] = []
    for b in bottles:
        h_px = int(b.bbox_xyxy[3]) - int(b.bbox_xyxy[1])
        if h_px < 10:
            continue
        # Depth from apparent height (pinhole)
        depth = _BOTTLE_REAL_HEIGHT_M * fy / h_px
        # Camera-frame 3-D (X right, Y down, Z forward)
        x_cam = (b.center_x - cx_px) * depth / fx
        z_cam = depth
        # Rough camera→world: camera is ~1 m in front of robot at 0.5 m height
        # mapping: world_x ≈ x_cam, world_y ≈ 1.0 - z_cam*0.8, world_z ≈ 0.3
        x_w = float(np.clip(x_cam, -0.8, 0.8))
        y_w = float(np.clip(1.0 - z_cam * 0.8, -0.5, 1.0))
        z_w = 0.30
        positions.append(np.array([x_w, y_w, z_w]))
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
# Argument parsing + main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Vision-Based Dynamic Obstacle Avoidance for UR5e"
    )
    p.add_argument(
        "--mode",
        choices=["sim", "sphere", "moving", "camera", "compare", "cv_test"],
        default="sim",
        help=(
            "sim     — cylindrical obstacle, NMPC avoidance (default)\n"
            "sphere  — spherical obstacle (r=10 cm), NMPC avoidance\n"
            "moving  — obstacle oscillates, tests dynamic re-planning\n"
            "camera  — live webcam YOLO+ArUco (requires cv2 + ultralytics)\n"
            "compare — headless benchmark: NMPC vs APF vs RRTConnect (saves 3 videos)\n"
            "cv_test — test vision pipeline: camera preview + MuJoCo, no NMPC"
        ),
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
    return p.parse_args()


def main():
    args = _parse_args()
    target = np.array(args.target, dtype=float)

    print(f"\n[MAIN] Mode    : {args.mode}")
    print(f"[MAIN] Target  : {target}")

    if args.mode == "compare":
        run_comparison(target)
    elif args.mode == "cv_test":
        run_cv_test(target)
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
