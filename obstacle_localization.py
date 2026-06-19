"""
obstacle_localization.py
------------------------
Converts bottle pixel detections into 3-D world positions using ArUco markers.

Reuses bottle_detector.py — does NOT reimplement detection.
The ArUco marker establishes the world frame; bottles are located by
back-projecting their pixel centre onto the z=BOTTLE_PLANE_Z plane
expressed in the marker (world) frame.

World frame convention (ArUco board origin):
    X  = right
    Y  = forward (camera looks toward +Y)
    Z  = up

Returns per-bottle:  obstacle_position = [x, y, z]  in metres.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import cv2

from config import (
    CAMERA_MATRIX, DIST_COEFFS, ARUCO_MARKER_SIZE_M,
    ARUCO_DICT_ID, BOTTLE_PLANE_Z,
)

# Import existing detection utilities from bottle_detector.py
from bottle_detector import (
    BottleDetection,
    ArucoDetection,
    DetectorConfig,
    build_capture,
    load_yolo,
    build_aruco_detector,
    parse_bottle_detections,
    detect_aruco,
)


# ---------------------------------------------------------------------------
# Pixel → world back-projection
# ---------------------------------------------------------------------------

def pixel_to_camera_ray(
    u: int,
    v: int,
    K: np.ndarray = CAMERA_MATRIX,
    dist: np.ndarray = DIST_COEFFS,
) -> np.ndarray:
    """
    Returns unit ray in camera frame for image point (u, v).
    Undistorts the pixel first.
    """
    pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
    undist = cv2.undistortPoints(pts, K, dist, P=K)
    u_u = float(undist[0, 0, 0])
    v_u = float(undist[0, 0, 1])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ray_cam = np.array([(u_u - cx) / fx, (v_u - cy) / fy, 1.0])
    return ray_cam / np.linalg.norm(ray_cam)


def ray_plane_intersect(
    ray_cam:   np.ndarray,   # (3,) unit ray in camera frame
    R_cam2world: np.ndarray, # (3,3) camera-to-world rotation
    t_cam2world: np.ndarray, # (3,) camera-to-world translation
    plane_z_world: float,    # Z height of intersection plane in world frame
) -> Optional[np.ndarray]:
    """
    Intersects a camera ray with the horizontal plane Z = plane_z_world
    in world coordinates.  Returns the 3-D world point or None if the
    ray is nearly parallel to the plane.
    """
    # Camera origin in world frame
    cam_origin_w = t_cam2world.flatten()
    # Ray direction in world frame
    ray_world    = R_cam2world @ ray_cam.flatten()

    # Parametric: P = cam_origin_w + t * ray_world,  P[2] = plane_z_world
    denom = ray_world[2]
    if abs(denom) < 1e-6:
        return None
    t = (plane_z_world - cam_origin_w[2]) / denom
    if t < 0:
        return None
    return cam_origin_w + t * ray_world


# ---------------------------------------------------------------------------
# Main conversion: detections → world positions
# ---------------------------------------------------------------------------

def bottles_to_world(
    bottles:  List[BottleDetection],
    markers:  List[ArucoDetection],
    plane_z:  float = BOTTLE_PLANE_Z,
    K:        np.ndarray = CAMERA_MATRIX,
    dist:     np.ndarray = DIST_COEFFS,
) -> List[np.ndarray]:
    """
    For each detected bottle, back-project its pixel centre to a 3-D world
    position using the first visible ArUco marker as the world frame origin.

    The ArUco solvePnP gives:
        R_cam2marker,  t_cam2marker  (marker frame expressed in camera frame)
    We invert to get:
        R_marker2cam = R_cam2marker.T
        t_marker2cam = -R_cam2marker.T @ t_cam2marker
    Then the camera origin in world (marker) frame:
        cam_pos_world = -R_cam2marker.T @ t_cam2marker
    And camera-to-world rotation:
        R_cam2world = R_cam2marker.T

    Returns:
        List of [x, y, z] world positions (one per bottle).
        Empty if no ArUco marker is visible.
    """
    if not markers or not bottles:
        return []

    # Use the first detected marker as world frame
    m = markers[0]
    R_marker2cam = m.rotation_matrix          # 3×3, marker expressed in camera
    t_marker2cam = m.tvec.flatten()           # camera→marker translation

    # Camera origin and axes in world (marker) frame
    R_cam2world = R_marker2cam.T
    t_cam2world = -R_marker2cam.T @ t_marker2cam   # camera origin in world

    world_positions: List[np.ndarray] = []
    for bottle in bottles:
        ray_cam = pixel_to_camera_ray(bottle.center_x, bottle.center_y, K, dist)
        p_world = ray_plane_intersect(ray_cam, R_cam2world, t_cam2world, plane_z)
        if p_world is not None:
            world_positions.append(p_world)

    return world_positions


# ---------------------------------------------------------------------------
# Live vision pipeline (uses existing bottle_detector infrastructure)
# ---------------------------------------------------------------------------

class VisionObstacleTracker:
    """
    Wraps YOLO + ArUco into a simple interface:
        tracker.update(frame) → List[np.ndarray]  (world positions)
    Designed to be called once per camera frame.
    """

    def __init__(self, cfg: DetectorConfig = None):
        self.cfg = cfg or DetectorConfig()
        self.model, self.bottle_class_id = load_yolo(self.cfg)
        self.aruco_detector = build_aruco_detector(self.cfg)
        self.last_bottles:    List[BottleDetection] = []
        self.last_markers:    List[ArucoDetection]  = []
        self.last_world_pos:  List[np.ndarray]      = []

    def update(self, frame: np.ndarray) -> List[np.ndarray]:
        """Process one BGR frame; return list of bottle world positions."""
        results = self.model.predict(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            verbose=False,
            device="cpu",
        )
        bottles = parse_bottle_detections(
            results[0], self.bottle_class_id, self.cfg.conf_threshold
        )
        markers = detect_aruco(frame, self.aruco_detector)

        world_pos = bottles_to_world(bottles, markers)

        self.last_bottles   = bottles
        self.last_markers   = markers
        self.last_world_pos = world_pos
        return world_pos

    def open_camera(self) -> "cv2.VideoCapture":
        return build_capture(self.cfg)
