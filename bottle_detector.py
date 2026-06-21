"""
bottle_detector.py
------------------
Real-time bottle detection (YOLOv8) + ArUco pose estimation (OpenCV).

Pipeline:
  VideoCapture -> YOLOv8 bottle detection
               -> ArUco marker detection + 6-DoF pose (rvec, tvec)
               -> draw overlays -> display + print pose for MuJoCo / UR5e

Output per ArUco marker (camera frame, metres):
  position  : tvec  [x, y, z]
  rotation  : rvec  (Rodrigues) -> rotation matrix -> quaternion (w,x,y,z)

Press 'q' to quit.

CAMERA CALIBRATION
------------------
Replace CAMERA_MATRIX and DIST_COEFFS below with values from your own
calibration (use opencv/calib3d checkerboard calibration or ROS camera_calibration).
The defaults are rough estimates for a 1280x720 webcam — pose numbers will
be inaccurate until you supply real calibration data.

ARUCO MARKER SIZE
-----------------
Set ARUCO_MARKER_SIZE_M to the physical side length of your printed marker
in metres (e.g. 0.05 for a 5 cm marker).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Camera intrinsics — REPLACE with your calibration values
# ---------------------------------------------------------------------------

# Approximate values for a generic 1280x720 webcam (focal length ~ 900 px)
CAMERA_MATRIX = np.array([
    [900.0,   0.0, 640.0],
    [  0.0, 900.0, 360.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)  # assume no distortion until calibrated

# Physical side length of the ArUco marker in metres
ARUCO_MARKER_SIZE_M: float = 0.05   # 5 cm — adjust to match your printed marker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """Runtime parameters — change here, nowhere else in the code."""
    # YOLOv8
    model_name: str        = "yolov8n.pt"
    conf_threshold: float  = 0.40
    iou_threshold: float   = 0.45
    target_class: str      = "bottle"

    # Camera
    camera_index: int  = 0
    frame_width: int   = 1280
    frame_height: int  = 720

    # ArUco — choose the dictionary that matches your printed markers
    # Common choices: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250
    aruco_dict_id: int = cv2.aruco.DICT_4X4_50


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BottleDetection:
    bbox_xyxy: np.ndarray   # [x1, y1, x2, y2] pixels
    confidence: float
    center_x: int
    center_y: int


@dataclass
class ArucoDetection:
    marker_id: int
    corners: np.ndarray      # shape (4, 2) — pixel corners
    rvec: np.ndarray         # Rodrigues rotation vector (3,)
    tvec: np.ndarray         # translation vector in metres (3,)
    rotation_matrix: np.ndarray   # 3x3 rotation matrix
    quaternion: np.ndarray   # (w, x, y, z) — ready for MuJoCo


# ---------------------------------------------------------------------------
# Camera / model setup
# ---------------------------------------------------------------------------

def build_capture(cfg: DetectorConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {cfg.camera_index}. "
            "Check that a webcam is connected and not in use by another app."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    return cap


def load_yolo(cfg: DetectorConfig) -> Tuple[YOLO, int]:
    model = YOLO(cfg.model_name)
    class_id = None
    for idx, name in model.names.items():
        if name.lower() == cfg.target_class.lower():
            class_id = idx
            break
    if class_id is None:
        raise ValueError(f"Class '{cfg.target_class}' not found in model.")
    print(f"[INFO] YOLO model loaded: {cfg.model_name}")
    print(f"[INFO] Target class: '{cfg.target_class}' (class_id={class_id})")
    return model, class_id


def build_aruco_detector(cfg: DetectorConfig) -> cv2.aruco.ArucoDetector:
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cfg.aruco_dict_id)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    print(f"[INFO] ArUco detector ready (dict_id={cfg.aruco_dict_id}, "
          f"marker_size={ARUCO_MARKER_SIZE_M*100:.0f} cm)")
    return detector


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def rvec_to_quaternion(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rotation vector to quaternion (w, x, y, z)."""
    R, _ = cv2.Rodrigues(rvec)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def parse_bottle_detections(
    result,
    bottle_class_id: int,
    conf_threshold: float,
) -> List[BottleDetection]:
    detections: List[BottleDetection] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf   = float(box.conf[0].item())
        if cls_id != bottle_class_id or conf < conf_threshold:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        detections.append(BottleDetection(
            bbox_xyxy  = np.array([x1, y1, x2, y2], dtype=int),
            confidence = conf,
            center_x   = (x1 + x2) // 2,
            center_y   = (y1 + y2) // 2,
        ))
    return detections


def detect_aruco(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
) -> List[ArucoDetection]:
    """Detect ArUco markers and estimate their 6-DoF pose via solvePnP."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = detector.detectMarkers(gray)

    results: List[ArucoDetection] = []
    if ids is None or len(ids) == 0:
        return results

    # 3-D corner coordinates of a flat marker centred at its origin
    half = ARUCO_MARKER_SIZE_M / 2.0
    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    for i, marker_id in enumerate(ids.flatten()):
        img_pts = corners_list[i].reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, CAMERA_MATRIX, DIST_COEFFS,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue
        rvec = rvec.flatten()
        tvec = tvec.flatten()
        R, _ = cv2.Rodrigues(rvec)
        quat = rvec_to_quaternion(rvec)
        results.append(ArucoDetection(
            marker_id       = int(marker_id),
            corners         = corners_list[i].reshape(4, 2),
            rvec            = rvec,
            tvec            = tvec,
            rotation_matrix = R,
            quaternion      = quat,
        ))
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_BOTTLE_BOX_COLOR  = (0, 255, 0)
_BOTTLE_TEXT_BG    = (0, 180, 0)
_CENTER_COLOR      = (0, 0, 255)
_ARUCO_COLOR       = (255, 165, 0)    # orange outline for ArUco
_FONT              = cv2.FONT_HERSHEY_SIMPLEX


def draw_bottle_detections(frame: np.ndarray, detections: List[BottleDetection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        cx, cy = det.center_x, det.center_y

        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOTTLE_BOX_COLOR, 2)

        label = f"bottle {det.confidence * 100:.1f}%"
        (tw, th), bl = cv2.getTextSize(label, _FONT, 0.6, 2)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, label_y - th - bl), (x1 + tw, label_y + bl),
                      _BOTTLE_TEXT_BG, cv2.FILLED)
        cv2.putText(frame, label, (x1, label_y), _FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.circle(frame, (cx, cy), 5, _CENTER_COLOR, cv2.FILLED)
        cv2.putText(frame, f"({cx},{cy})", (cx + 7, cy + 5),
                    _FONT, 0.45, _CENTER_COLOR, 1, cv2.LINE_AA)


def draw_aruco_detections(frame: np.ndarray, detections: List[ArucoDetection]) -> None:
    for det in detections:
        # Draw marker border
        pts = det.corners.astype(int)
        cv2.polylines(frame, [pts], isClosed=True, color=_ARUCO_COLOR, thickness=2)

        # Draw XYZ axes at marker origin (axis length = half marker size)
        cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS,
                          det.rvec, det.tvec, ARUCO_MARKER_SIZE_M * 0.5)

        # Label: marker ID + z-distance
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        z_cm = det.tvec[2] * 100.0
        id_label = f"ID:{det.marker_id}  z={z_cm:.1f}cm"
        cv2.putText(frame, id_label, (cx - 40, cy - 10),
                    _FONT, 0.55, _ARUCO_COLOR, 2, cv2.LINE_AA)


def draw_stats(frame: np.ndarray, fps: float, n_bottles: int, n_markers: int) -> None:
    cv2.putText(frame,
                f"FPS:{fps:.1f}  Bottles:{n_bottles}  ArUco:{n_markers}",
                (10, 30), _FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Terminal print helpers
# ---------------------------------------------------------------------------

def print_bottle(i: int, det: BottleDetection) -> None:
    print(
        f"[BOTTLE {i}] conf={det.confidence:.2f}  "
        f"center=({det.center_x},{det.center_y})  "
        f"bbox=[{det.bbox_xyxy[0]},{det.bbox_xyxy[1]},"
        f"{det.bbox_xyxy[2]},{det.bbox_xyxy[3]}]"
    )


def print_aruco(det: ArucoDetection) -> None:
    tx, ty, tz = det.tvec
    w, qx, qy, qz = det.quaternion
    print(
        f"[ARUCO ID:{det.marker_id}] "
        f"pos=({tx:.4f}, {ty:.4f}, {tz:.4f}) m  "
        f"quat=(w={w:.4f}, x={qx:.4f}, y={qy:.4f}, z={qz:.4f})"
    )
    # Compact MuJoCo-ready format: <body pos="tx ty tz" quat="w qx qy qz">
    print(
        f"  -> MuJoCo: pos=\"{tx:.4f} {ty:.4f} {tz:.4f}\"  "
        f"quat=\"{w:.4f} {qx:.4f} {qy:.4f} {qz:.4f}\""
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg: DetectorConfig) -> None:
    cap      = build_capture(cfg)
    model, bottle_class_id = load_yolo(cfg)
    aruco_detector         = build_aruco_detector(cfg)

    # Warm-up inference
    dummy = np.zeros((cfg.frame_height, cfg.frame_width, 3), dtype=np.uint8)
    model.predict(dummy, conf=cfg.conf_threshold, iou=cfg.iou_threshold, verbose=False)

    print("\n[INFO] Detection loop started.  Press 'q' to quit.\n")
    print("NOTE: Calibrate your camera and update CAMERA_MATRIX / DIST_COEFFS")
    print("      for accurate ArUco pose (tvec in metres).\n")

    prev_tick = time.perf_counter()
    fps = 0.0

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Failed to read frame. Retrying...")
            continue

        frame = cv2.flip(frame, 1)  # mirror horizontally

        frame_count += 1
        # Debug: save first 3 frames to check if camera data is real
        if frame_count <= 3:
            cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)
            print(f"[DEBUG] Frame {frame_count}: shape={frame.shape} dtype={frame.dtype} "
                  f"min={frame.min()} max={frame.max()}")

        # --- YOLO bottle detection ---
        results = model.predict(
            frame, conf=cfg.conf_threshold, iou=cfg.iou_threshold,
            verbose=False, device="cpu",
        )
        bottles = parse_bottle_detections(results[0], bottle_class_id, cfg.conf_threshold)

        # --- ArUco detection + pose ---
        markers = detect_aruco(frame, aruco_detector)

        # --- Terminal output ---
        for i, b in enumerate(bottles, 1):
            print_bottle(i, b)
        for m in markers:
            print_aruco(m)

        # --- Draw overlays ---
        draw_bottle_detections(frame, bottles)
        draw_aruco_detections(frame, markers)

        # FPS (EMA)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_tick, 1e-6))
        prev_tick = now

        draw_stats(frame, fps, len(bottles), len(markers))

        cv2.imshow("Bottle + ArUco Detector  [q = quit]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] 'q' pressed — exiting.")
            break
        if cv2.getWindowProperty("Bottle + ArUco Detector  [q = quit]",
                                 cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Window closed — exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = DetectorConfig()
    if len(sys.argv) > 1:
        try:
            cfg.camera_index = int(sys.argv[1])
        except ValueError:
            print(f"[ERROR] Invalid camera index '{sys.argv[1]}'. Using default 0.")
    run(cfg)
