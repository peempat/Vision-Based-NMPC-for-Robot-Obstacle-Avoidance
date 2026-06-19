"""
evaluation.py
-------------
Records and summarises performance metrics during a simulation episode.

Metrics:
    tracking_error      — running RMS of ee-to-target distance
    min_clearance       — minimum obstacle clearance observed
    mean_solve_time     — average NMPC solve time
    max_solve_time      — worst-case NMPC solve time
    success             — True if goal reached without constraint violation
    constraint_violations — count of frames where clearance < 0
    path_length         — total distance travelled by end-effector [m]
    end_effector_error  — final distance from ee to target [m]

Full trajectory data (for plotting) is stored in:
    ee_positions   — list of np.ndarray (3,)
    timestamps     — list of float (sim time [s])
    q_dot_norms    — list of float (joint velocity norm)
    clearances     — list of float
    solve_times    — list of float [s]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class EpisodeEvaluator:
    # ---- scalar time-series ----
    tracking_errors:   List[float]       = field(default_factory=list)
    clearances:        List[float]       = field(default_factory=list)
    solve_times:       List[float]       = field(default_factory=list)
    q_dot_norms:       List[float]       = field(default_factory=list)
    solver_statuses:   List[str]         = field(default_factory=list)
    timestamps:        List[float]       = field(default_factory=list)

    # ---- trajectory ----
    ee_positions:      List[np.ndarray]  = field(default_factory=list)  # (3,) each

    # ---- aggregates ----
    violations:        int               = 0
    path_length:       float             = 0.0

    # ---- bookkeeping ----
    last_ee_pos:       Optional[np.ndarray] = None
    target_pos:        Optional[np.ndarray] = None

    def record(
        self,
        sim_time:     float,
        ee_pos:       np.ndarray,
        target_pos:   np.ndarray,
        q_dot:        np.ndarray,
        clearance:    float,
        solve_time:   float,
        solver_status: str,
    ) -> None:
        dist = float(np.linalg.norm(ee_pos - target_pos))
        self.tracking_errors.append(dist)
        self.clearances.append(clearance)
        self.solve_times.append(solve_time)
        self.q_dot_norms.append(float(np.linalg.norm(q_dot)))
        self.solver_statuses.append(solver_status)
        self.timestamps.append(sim_time)
        self.ee_positions.append(ee_pos.copy())

        if self.target_pos is None:
            self.target_pos = target_pos.copy()

        if clearance < 0:
            self.violations += 1

        if self.last_ee_pos is not None:
            self.path_length += float(np.linalg.norm(ee_pos - self.last_ee_pos))
        self.last_ee_pos = ee_pos.copy()

    def summary(self) -> dict:
        if not self.tracking_errors:
            return {}
        return {
            "tracking_rms_m":        float(np.sqrt(np.mean(np.square(self.tracking_errors)))),
            "tracking_mean_m":       float(np.mean(self.tracking_errors)),
            "min_clearance_m":       float(np.min(self.clearances)),
            "mean_clearance_m":      float(np.mean(self.clearances)),
            "mean_solve_time_ms":    float(np.mean(self.solve_times)) * 1e3,
            "max_solve_time_ms":     float(np.max(self.solve_times)) * 1e3,
            "constraint_violations": self.violations,
            "path_length_m":         self.path_length,
            "end_effector_error_m":  float(self.tracking_errors[-1]),
            "n_steps":               len(self.tracking_errors),
            "success":               (
                self.violations == 0
                and self.tracking_errors[-1] < 0.05
            ),
            # Full trajectory data (for plotting)
            "trajectory": {
                "timestamps":   list(self.timestamps),
                "ee_positions": [p.tolist() for p in self.ee_positions],
                "clearances":   list(self.clearances),
                "solve_times":  [t * 1e3 for t in self.solve_times],  # ms
                "q_dot_norms":  list(self.q_dot_norms),
                "errors":       list(self.tracking_errors),
                "target_pos":   self.target_pos.tolist() if self.target_pos is not None else None,
            },
        }

    def print_summary(self) -> None:
        s = self.summary()
        if not s:
            print("[EVAL] No data recorded.")
            return
        print("\n" + "=" * 52)
        print("  Episode Evaluation Summary")
        print("=" * 52)
        skip = {"trajectory"}
        for k, v in s.items():
            if k in skip:
                continue
            if isinstance(v, float):
                print(f"  {k:<32s}: {v:.4f}")
            else:
                print(f"  {k:<32s}: {v}")
        print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# Multi-episode comparison table
# ---------------------------------------------------------------------------

def compare_methods(results: dict[str, dict]) -> None:
    metrics = [
        "tracking_rms_m",
        "min_clearance_m",
        "path_length_m",
        "mean_solve_time_ms",
        "constraint_violations",
        "success",
    ]
    col_w = 20
    header = f"{'Metric':<28s}" + "".join(f"{m:>{col_w}s}" for m in results)
    print("\n" + "=" * (28 + col_w * len(results)))
    print("  Method Comparison")
    print("=" * (28 + col_w * len(results)))
    print(header)
    print("-" * (28 + col_w * len(results)))
    for m in metrics:
        row = f"{m:<28s}"
        for method_res in results.values():
            val = method_res.get(m, "N/A")
            if isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{str(val):>{col_w}s}"
        print(row)
    print("=" * (28 + col_w * len(results)) + "\n")
