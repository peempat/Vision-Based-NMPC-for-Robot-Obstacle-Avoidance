"""
plot_results.py
---------------
Generate trajectory plots and save video from a completed episode.

Outputs (saved to results/ directory):
    trajectory_3d.png   — 3-D end-effector path with obstacles and target
    metrics.png         — 4-panel: tracking error / clearance / solve time / joint speed
    simulation.gif      — rendered MuJoCo simulation video (Pillow GIF)
"""
from __future__ import annotations

import os
import datetime
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # off-screen backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers 3-D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

# ── output directory ────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "results")
os.makedirs(_RESULTS, exist_ok=True)


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# 3-D trajectory plot
# ============================================================================

def plot_trajectory_3d(
    trajectory:         dict,                    # from EpisodeEvaluator.summary()["trajectory"]
    obstacle_positions: List[np.ndarray],
    obstacle_radius:    float,
    safety_margin:      float,
    filename:           str  = None,
) -> str:
    ee_pos    = np.array(trajectory["ee_positions"])   # (N, 3)
    target    = np.array(trajectory["target_pos"])
    errors    = np.array(trajectory["errors"])         # colour by error magnitude

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # ── colour the path by tracking error (red=far, green=close) ────────────
    norm = plt.Normalize(vmin=0, vmax=errors.max())
    cmap = plt.cm.RdYlGn_r

    for i in range(len(ee_pos) - 1):
        seg   = ee_pos[i : i + 2]
        color = cmap(norm((errors[i] + errors[i + 1]) / 2))
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                color=color, linewidth=2.5, solid_capstyle="round")

    # ── start / goal markers ─────────────────────────────────────────────────
    ax.scatter(*ee_pos[0],  s=120, c="royalblue", marker="o",
               zorder=5, label="Start")
    ax.scatter(*ee_pos[-1], s=250, c="limegreen", marker="*",
               zorder=5, label="End")
    ax.scatter(*target,     s=200, c="lime",       marker="^",
               zorder=5, label="Target")

    # ── obstacles (wireframe spheres) ─────────────────────────────────────────
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi,     16)
    for k, obs_pos in enumerate(obstacle_positions):
        r  = obstacle_radius
        xs = obs_pos[0] + r * np.outer(np.cos(u), np.sin(v))
        ys = obs_pos[1] + r * np.outer(np.sin(u), np.sin(v))
        zs = obs_pos[2] + r * np.outer(np.ones(u.size), np.cos(v))
        ax.plot_surface(xs, ys, zs, color="tomato", alpha=0.35)
        ax.plot_wireframe(xs, ys, zs, color="red", linewidth=0.4, alpha=0.5)

        # Safety margin bubble
        rs = obstacle_radius + safety_margin
        xs2 = obs_pos[0] + rs * np.outer(np.cos(u), np.sin(v))
        ys2 = obs_pos[1] + rs * np.outer(np.sin(u), np.sin(v))
        zs2 = obs_pos[2] + rs * np.outer(np.ones(u.size), np.cos(v))
        ax.plot_wireframe(xs2, ys2, zs2, color="orange", linewidth=0.3,
                          alpha=0.2, label=f"Safety zone" if k == 0 else None)

        ax.text(obs_pos[0], obs_pos[1], obs_pos[2] + obstacle_radius + 0.03,
                f"Obs {k}", color="red", fontsize=8, ha="center")

    # ── colorbar ─────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.1)
    cbar.set_label("EE distance to target [m]", fontsize=9)

    ax.set_xlabel("X [m]", fontsize=9)
    ax.set_ylabel("Y [m]", fontsize=9)
    ax.set_zlabel("Z [m]", fontsize=9)
    ax.set_title("UR5e End-Effector Trajectory\n"
                 "Kinematic NMPC with Obstacle Avoidance", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    path = filename or os.path.join(_RESULTS, f"trajectory_3d_{_stamp()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] 3-D trajectory saved → {path}")
    return path


# ============================================================================
# 4-panel metrics plot
# ============================================================================

def plot_metrics(
    trajectory:  dict,
    safety_margin: float,
    filename:    str = None,
) -> str:
    ts        = np.array(trajectory["timestamps"])
    errors    = np.array(trajectory["errors"])
    clearance = np.array(trajectory["clearances"])
    solve_ms  = np.array(trajectory["solve_times"])    # already in ms
    qd_norms  = np.array(trajectory["q_dot_norms"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("NMPC Episode Metrics — UR5e Obstacle Avoidance", fontsize=13)

    # ── (0,0) Tracking error ──────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(ts, errors, color="steelblue", linewidth=2, label="EE error")
    ax.fill_between(ts, 0, errors, alpha=0.15, color="steelblue")
    ax.axhline(0.02, color="green", linestyle="--", linewidth=1, label="Goal tol (2 cm)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance to target [m]")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── (0,1) Obstacle clearance ──────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(ts, clearance, color="darkorange", linewidth=2, label="Min clearance")
    ax.fill_between(ts, 0, clearance, alpha=0.15, color="darkorange")
    ax.axhline(safety_margin, color="red", linestyle="--", linewidth=1.5,
               label=f"Safety margin ({safety_margin:.2f} m)")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    # shade violation zone
    ax.fill_between(ts, 0, safety_margin,
                    where=(clearance < safety_margin),
                    alpha=0.25, color="red", label="Margin violated")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Clearance [m]")
    ax.set_title("Obstacle Clearance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,0) NMPC solve time ─────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.bar(ts, solve_ms, width=np.diff(ts, append=ts[-1] + (ts[-1]-ts[-2])) * 0.8,
           color="mediumpurple", alpha=0.8, align="edge")
    ax.axhline(200, color="red", linestyle="--", linewidth=1, label="200 ms budget")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Solve time [ms]")
    ax.set_title(f"NMPC Solve Time  (mean {solve_ms.mean():.0f} ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)

    # ── (1,1) Joint velocity norm ─────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(ts, qd_norms, color="teal", linewidth=2, label="‖q̇‖")
    ax.fill_between(ts, 0, qd_norms, alpha=0.15, color="teal")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint velocity norm [rad/s]")
    ax.set_title("Control Effort")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = filename or os.path.join(_RESULTS, f"metrics_{_stamp()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Metrics plot saved → {path}")
    return path


# ============================================================================
# Video saving  (Pillow animated GIF)
# ============================================================================

def save_video(
    frames:   List[np.ndarray],   # list of (H, W, 3) uint8 arrays
    filename: str = None,
    fps:      int = 10,
) -> str:
    if not frames:
        print("[VIDEO] No frames to save.")
        return ""
    path = filename or os.path.join(_RESULTS, f"simulation_{_stamp()}.gif")
    images = [Image.fromarray(f) for f in frames]
    # Quantize each frame to 256 colours (required for GIF)
    images_quantized = [img.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                        for img in images]
    images_quantized[0].save(
        path,
        save_all      = True,
        append_images = images_quantized[1:],
        duration      = int(1000 / fps),
        loop          = 0,
        optimize      = False,
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"[VIDEO] Saved {len(frames)} frames @ {fps} fps → {path}  ({size_mb:.1f} MB)")
    return path


# ============================================================================
# Convenience: run everything after an episode
# ============================================================================

def save_all(
    results:            dict,               # from EpisodeEvaluator.summary()
    video_frames:       List[np.ndarray],
    obstacle_positions: List[np.ndarray],
    obstacle_radius:    float,
    safety_margin:      float,
    fps:                int = 10,
    tag:                str = "",
) -> dict[str, str]:
    """
    Save 3-D trajectory plot, metrics plot, and GIF video.
    Returns dict mapping artifact name → file path.
    """
    stamp = _stamp() + (f"_{tag}" if tag else "")
    traj  = results.get("trajectory", {})

    paths = {}

    if traj:
        paths["trajectory_3d"] = plot_trajectory_3d(
            trajectory         = traj,
            obstacle_positions = obstacle_positions,
            obstacle_radius    = obstacle_radius,
            safety_margin      = safety_margin,
            filename           = os.path.join(_RESULTS, f"trajectory_3d_{stamp}.png"),
        )
        paths["metrics"] = plot_metrics(
            trajectory    = traj,
            safety_margin = safety_margin,
            filename      = os.path.join(_RESULTS, f"metrics_{stamp}.png"),
        )

    if video_frames:
        paths["video"] = save_video(
            frames   = video_frames,
            filename = os.path.join(_RESULTS, f"simulation_{stamp}.gif"),
            fps      = fps,
        )

    return paths
