---
name: project-bottle-detector
description: Real-time bottle detection app written for Panuwit — YOLOv8n + OpenCV webcam pipeline targeting CPU inference
metadata:
  type: project
---

A standalone real-time detection script `bottle_detector.py` was created in C:\Users\Panuwit\Downloads\cvforRobot\.

Key design decisions made:
- Model: yolov8n.pt (nano) — best CPU throughput; user explicitly needed CPU-only operation.
- Class filtering done post-inference by matching the integer class ID for "bottle" against model.names, making it robust to model swaps.
- DetectorConfig dataclass centralises all tuneable parameters (conf_threshold=0.40, iou=0.45, resolution 1280x720).
- One warm-up inference pass on a dummy frame to avoid first-frame latency spike.
- EMA-smoothed FPS display (alpha=0.9) to avoid jitter.
- Window-close detection via cv2.WND_PROP_VISIBLE so users can quit with the OS X button, not just 'q'.
- Terminal output: one line per detected bottle per frame with conf, center (cx,cy), and bbox.

**Why:** User wanted a clean, well-commented starter script for robotics CV experimentation on CPU hardware.
**How to apply:** If extending this project, keep the DetectorConfig dataclass pattern; add new fields there rather than hardcoding values in functions.
