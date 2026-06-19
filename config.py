"""
config.py
---------
All tunable parameters for the Vision-Based Dynamic Obstacle Avoidance system.
Edit here; do not scatter magic numbers across other modules.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

# ---------------------------------------------------------------------------
# Camera / ArUco
# ---------------------------------------------------------------------------

CAMERA_MATRIX = np.array([
    [900.0,   0.0, 640.0],
    [  0.0, 900.0, 360.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)

ARUCO_MARKER_SIZE_M: float = 0.05   # physical side length of printed marker [m]
ARUCO_DICT_ID: int = 4              # cv2.aruco.DICT_4X4_50

# Height of bottle base above the ArUco board plane [m].
# Used when back-projecting pixel coords to world.
BOTTLE_PLANE_Z: float = 0.0


# ---------------------------------------------------------------------------
# Obstacle / bottle geometry
# ---------------------------------------------------------------------------

@dataclass
class ObstacleConfig:
    radius: float        = 0.07    # obstacle sphere radius [m]  (larger = more visible avoidance)
    height: float        = 0.30    # obstacle height [m]
    safety_margin: float = 0.10    # extra clearance beyond radius [m]
    shape: str           = "sphere"


# ---------------------------------------------------------------------------
# UR5e kinematics (DH, standard convention)
# ---------------------------------------------------------------------------

# a [m], d [m], alpha [rad]  — 6 joints
UR5E_A     = [0.0,    -0.425,  -0.3922, 0.0,    0.0,    0.0   ]
UR5E_D     = [0.1625,  0.0,     0.0,    0.1333, 0.0997, 0.0996]
UR5E_ALPHA = [np.pi/2, 0.0,     0.0,    np.pi/2, -np.pi/2, 0.0]

# Joint limits [rad]  (size3: ±2π; elbow size3_limited: ±π; wrists size1: ±2π)
UR5E_Q_MIN = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
UR5E_Q_MAX = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])

# Max joint velocity [rad/s] — conservative for safety
UR5E_QDOT_MAX = np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0])

# Link capsule radii [m] — one per link (shoulder … wrist_3)
UR5E_LINK_RADII = [0.060, 0.060, 0.055, 0.040, 0.040, 0.040]

# Home configuration matching the keyframe in ur5e.xml
UR5E_HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


# ---------------------------------------------------------------------------
# NMPC
# ---------------------------------------------------------------------------

@dataclass
class NMPCConfig:
    N:  int   = 10      # prediction horizon steps
    dt: float = 0.05   # time step per horizon step [s]

    # Cost weights
    w_tracking:   float = 100.0   # end-effector position error
    w_ctrl:       float = 0.05    # control effort  ||q_dot||^2
    w_smooth:     float = 0.3     # smoothness  ||Δq_dot||^2
    w_obstacle:   float = 1000.0  # obstacle soft-penalty weight (high = hard avoidance)
    w_terminal:   float = 300.0   # terminal tracking weight

    # Solver — tighter tolerances slow convergence; loosen for real-time
    ipopt_max_iter:  int  = 80
    ipopt_tol:       float = 5e-4
    ipopt_print:     int  = 0      # 0 = silent


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    max_obstacles: int  = 5        # pre-allocated obstacle bodies in scene
    sim_dt:        float = 0.002   # MuJoCo physics timestep [s]
    control_freq:  int  = 5        # Hz — NMPC runs at ~3-5 Hz (solver takes ~200-400 ms)
    render:        bool = True

    model_dir: str = "mujoco_menagerie main universal_robots_ur5e"
    scene_xml: str = "scene_obstacle_avoidance.xml"  # generated at runtime


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    apf_alpha:      float = 1.0    # attractive gain
    apf_beta:       float = 0.5    # repulsive gain
    apf_d0:         float = 0.3    # influence distance [m]
    apf_dt:         float = 0.05   # step size

    rrt_max_iter:   int   = 5000
    rrt_step_size:  float = 0.2    # [rad]
    rrt_goal_tol:   float = 0.05   # [m] end-effector goal tolerance
