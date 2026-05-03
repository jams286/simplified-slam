"""
Real-time SLAM with physical CSPC LiDAR sensor.
Runs EKF-SLAM using scan matching (ICP) for motion estimation.

No pre-existing map needed — this IS the point of SLAM:
build the map while simultaneously localizing yourself.

Usage:
    python main_real.py --port COM3 --version 2
    python main_real.py --port COM3 --version 2 --baud 230400 --max-range 5.0
"""

import numpy as np
import sys
import os
import argparse
import time
import signal
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import deque

from config import (
    GRID_RESOLUTION, GRID_LOG_ODD_FREE, GRID_LOG_ODD_OCC,
    GRID_LOG_ODD_PRIOR, GRID_LOG_ODD_MAX, GRID_LOG_ODD_MIN,
    EKF_RANGE_NOISE, EKF_BEARING_NOISE, EKF_ASSOCIATION_THRESHOLD,
)
from src.real_lidar import RealLiDAR
from src.scan_matching import ScanMatcher
from src.ekf_slam import EKFSLAM
from src.occupancy_grid import OccupancyGrid
from src.lidar import LiDAR, LiDARScan


# ─── Global stop flag ─────────────────────────────────────────────────────────
_running = True


def signal_handler(sig, frame):
    global _running
    print("\n⛔ Stopping SLAM...")
    _running = False


signal.signal(signal.SIGINT, signal_handler)


def extract_landmarks_from_scan(scan: LiDARScan,
                                 min_cluster_gap: float = 0.3) -> list:
    """
    Extract landmarks (corners/discontinuities) from a real LiDAR scan.
    Returns list of (range, angle) tuples.
    """
    landmarks = []
    ranges = scan.ranges
    valid = scan.valid

    for i in range(1, len(ranges) - 1):
        if not valid[i]:
            continue
        if valid[i-1] and valid[i+1]:
            diff_left = abs(ranges[i] - ranges[i-1])
            diff_right = abs(ranges[i] - ranges[i+1])
            if diff_left > min_cluster_gap or diff_right > min_cluster_gap:
                landmarks.append((ranges[i], scan.angles[i]))

    return landmarks


class RealTimeSLAMVisualizer:
    """
    Real-time matplotlib visualization for live SLAM.
    Shows: occupancy grid, robot trail, landmarks, and current scan.
    """

    def __init__(self, grid_width: float, grid_height: float,
                 max_range: float = 8.0):
        plt.ion()
        self.fig, (self.ax_scan, self.ax_map) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.suptitle("Real-Time SLAM — CSPC LiDAR", fontsize=14, fontweight='bold')
        self.max_range = max_range
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Scan view (polar-like in cartesian)
        self.ax_scan.set_xlim(-max_range, max_range)
        self.ax_scan.set_ylim(-max_range, max_range)
        self.ax_scan.set_aspect('equal')
        self.ax_scan.set_title("Current LiDAR Scan (local frame)")
        self.ax_scan.set_xlabel("X (m)")
        self.ax_scan.set_ylabel("Y (m)")
        self.ax_scan.grid(True, alpha=0.3)

        # Map view
        self.ax_map.set_xlim(0, grid_width)
        self.ax_map.set_ylim(0, grid_height)
        self.ax_map.set_aspect('equal')
        self.ax_map.set_title("Occupancy Grid Map")
        self.ax_map.set_xlabel("X (m)")
        self.ax_map.set_ylabel("Y (m)")

        self.fig.tight_layout()
        plt.pause(0.01)

    def update(self, scan_points_local: np.ndarray,
               occupancy_grid: OccupancyGrid,
               robot_pose: np.ndarray,
               trail: list,
               landmarks: np.ndarray,
               step: int,
               scan_freq: float = 0.0):
        """Update both panels with new data."""
        # --- Left panel: current scan ---
        self.ax_scan.clear()
        self.ax_scan.set_xlim(-self.max_range, self.max_range)
        self.ax_scan.set_ylim(-self.max_range, self.max_range)
        self.ax_scan.set_aspect('equal')
        self.ax_scan.set_title(f"LiDAR Scan (step {step}, {scan_freq:.0f} Hz)")
        self.ax_scan.grid(True, alpha=0.3)

        if len(scan_points_local) > 0:
            self.ax_scan.scatter(scan_points_local[:, 0],
                               scan_points_local[:, 1],
                               c='red', s=2, alpha=0.7)

        # Robot at origin with heading arrow
        self.ax_scan.plot(0, 0, 'bo', markersize=8)
        self.ax_scan.arrow(0, 0, 0.5, 0, head_width=0.15,
                          head_length=0.1, fc='blue', ec='blue')

        # --- Right panel: occupancy map ---
        self.ax_map.clear()
        self.ax_map.set_xlim(0, self.grid_width)
        self.ax_map.set_ylim(0, self.grid_height)
        self.ax_map.set_aspect('equal')
        coverage = np.mean(occupancy_grid.get_explored_mask())
        self.ax_map.set_title(f"Map ({coverage:.0%} explored, {len(landmarks)} landmarks)")
        self.ax_map.set_xlabel("X (m)")
        self.ax_map.set_ylabel("Y (m)")

        # Draw occupancy grid
        prob_map = occupancy_grid.get_probability_map()
        self.ax_map.imshow(prob_map, origin='lower', cmap='gray_r',
                          extent=[0, self.grid_width, 0, self.grid_height],
                          vmin=0, vmax=1, alpha=0.8)

        # Draw trail
        if len(trail) > 1:
            trail_arr = np.array(trail)
            self.ax_map.plot(trail_arr[:, 0], trail_arr[:, 1],
                           'b-', linewidth=1, alpha=0.6, label='Path')

        # Draw landmarks
        if len(landmarks) > 0:
            self.ax_map.scatter(landmarks[:, 0], landmarks[:, 1],
                              c='green', marker='^', s=40, alpha=0.8,
                              label='Landmarks')

        # Draw robot position
        rx, ry, rtheta = robot_pose
        self.ax_map.plot(rx, ry, 'bo', markersize=8)
        dx = 0.4 * np.cos(rtheta)
        dy = 0.4 * np.sin(rtheta)
        self.ax_map.arrow(rx, ry, dx, dy, head_width=0.15,
                         head_length=0.1, fc='blue', ec='blue')

        self.ax_map.legend(loc='upper right', fontsize=8)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


def run_real_slam(port: str = "COM3", version: int = 2,
                  baud_rate: int = None, max_range: float = 8.0,
                  min_range: float = 0.10, grid_size: float = 20.0,
                  grid_resolution: float = 0.1, max_steps: int = 0):
    """
    Run SLAM with a real CSPC LiDAR sensor.
    
    Args:
        port: Serial port (e.g., COM3, /dev/ttyUSB0)
        version: LiDAR version (1-4)
        baud_rate: Serial baud rate (None = auto-detect from version)
        max_range: Maximum range to consider (meters)
        min_range: Minimum range to consider (meters)
        grid_size: Map size in meters (square)
        grid_resolution: Occupancy grid resolution (m/cell)
        max_steps: Maximum scans to process (0 = unlimited)
    """
    global _running

    print("=" * 60)
    print("  REAL-TIME SLAM — CSPC LiDAR")
    print("=" * 60)
    print(f"  Port: {port}")
    print(f"  Version: {version}")
    print(f"  Baud: {baud_rate or 'auto'}")
    print(f"  Range: {min_range}–{max_range} m")
    print(f"  Map: {grid_size}x{grid_size} m @ {grid_resolution} m/cell")
    print("=" * 60)
    print("\n  Press Ctrl+C to stop.\n")

    # === Initialize components ===

    # Real LiDAR
    lidar = RealLiDAR(
        port=port, version=version, baud_rate=baud_rate,
        max_range=max_range, min_range=min_range, num_beams=360
    )

    # Scan matcher (ICP) — replaces odometry
    scan_matcher = ScanMatcher(
        max_iterations=50,
        tolerance=1e-4,
        max_correspondence_dist=0.5,
        min_points=30
    )

    # EKF-SLAM — robot starts at the center of the map
    initial_pose = np.array([grid_size / 2, grid_size / 2, 0.0])
    ekf = EKFSLAM(
        initial_pose=initial_pose,
        range_noise=EKF_RANGE_NOISE,
        bearing_noise=EKF_BEARING_NOISE,
        association_threshold=EKF_ASSOCIATION_THRESHOLD
    )

    # Occupancy grid
    occ_grid = OccupancyGrid(
        width=grid_size, height=grid_size, resolution=grid_resolution,
        log_odd_free=GRID_LOG_ODD_FREE, log_odd_occ=GRID_LOG_ODD_OCC,
        log_odd_prior=GRID_LOG_ODD_PRIOR, log_odd_max=GRID_LOG_ODD_MAX,
        log_odd_min=GRID_LOG_ODD_MIN
    )

    # Visualizer
    viz = RealTimeSLAMVisualizer(grid_size, grid_size, max_range)

    # State
    prev_points_local = None
    trail = [initial_pose[:2].tolist()]
    step = 0
    dt_estimate = 0.1  # Will be updated from scan frequency

    print("Connecting to LiDAR...")
    try:
        lidar.start()
    except Exception as e:
        print(f"ERROR: Could not connect to LiDAR on {port}: {e}")
        print("  - Check the port in Device Manager")
        print("  - Make sure no other program is using the port")
        print("  - Verify the LiDAR is powered and spinning")
        return

    print("LiDAR connected! Starting SLAM loop...\n")

    try:
        for scan in lidar.iter_scans():
            if not _running:
                break
            if max_steps > 0 and step >= max_steps:
                break

            step += 1

            # Get local points for scan matching
            points_local = lidar.scan_to_local_points(scan)

            if len(points_local) < 30:
                continue  # Skip sparse scans

            # === Motion estimation via scan matching ===
            if prev_points_local is not None:
                motion, icp_error = scan_matcher.match(
                    reference=prev_points_local,
                    current=points_local
                )

                # Convert ICP motion to velocity estimates
                # (for EKF prediction step)
                dx, dy, dtheta = motion
                linear_dist = np.sqrt(dx**2 + dy**2)

                # Estimate dt from scan frequency (or use fixed)
                v_estimate = linear_dist / dt_estimate
                omega_estimate = dtheta / dt_estimate

                # Limit unreasonable values (noise rejection)
                v_estimate = np.clip(v_estimate, -2.0, 2.0)
                omega_estimate = np.clip(omega_estimate, -np.pi, np.pi)
            else:
                v_estimate = 0.0
                omega_estimate = 0.0

            prev_points_local = points_local

            # === Update pose in the scan for map building ===
            estimated_pose = ekf.robot_pose
            scan_with_pose = LiDARScan(
                ranges=scan.ranges,
                angles=np.linspace(0, 2 * np.pi, len(scan.ranges), endpoint=False) + estimated_pose[2],
                valid=scan.valid,
                robot_pose=tuple(estimated_pose),
                max_range=max_range
            )

            # === Extract landmarks ===
            landmark_obs = extract_landmarks_from_scan(scan_with_pose)

            # === EKF-SLAM step ===
            ekf.step(v_estimate, omega_estimate, dt_estimate, landmark_obs)

            # === Update occupancy grid ===
            estimated_pose = ekf.robot_pose
            occ_grid.update_from_pose(scan_with_pose, estimated_pose)

            # === Record trail ===
            trail.append(estimated_pose[:2].tolist())

            # === Visualize (every 3rd scan for performance) ===
            if step % 3 == 0:
                viz.update(
                    scan_points_local=points_local,
                    occupancy_grid=occ_grid,
                    robot_pose=estimated_pose,
                    trail=trail,
                    landmarks=ekf.landmarks,
                    step=step,
                    scan_freq=1.0 / dt_estimate if dt_estimate > 0 else 0
                )

            # Progress print
            if step % 30 == 0:
                coverage = np.mean(occ_grid.get_explored_mask())
                print(f"  Step {step:5d} | Pose: ({estimated_pose[0]:.2f}, "
                      f"{estimated_pose[1]:.2f}, {np.degrees(estimated_pose[2]):.0f}°) | "
                      f"Landmarks: {ekf.num_landmarks} | Coverage: {coverage:.1%}")

    except Exception as e:
        print(f"\nError during SLAM: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nStopping LiDAR...")
        lidar.stop()
        viz.close()

    # === Save results ===
    print("\n" + "=" * 60)
    print("  SLAM SESSION COMPLETE")
    print("=" * 60)
    print(f"  Total scans processed: {step}")
    print(f"  Landmarks discovered: {ekf.num_landmarks}")
    coverage = np.mean(occ_grid.get_explored_mask())
    print(f"  Map coverage: {coverage:.1%}")
    print(f"  Final pose: ({estimated_pose[0]:.2f}, {estimated_pose[1]:.2f}, "
          f"{np.degrees(estimated_pose[2]):.0f}°)")

    # Save map
    os.makedirs("results", exist_ok=True)
    prob_map = occ_grid.get_probability_map()
    plt.figure(figsize=(10, 10))
    plt.imshow(prob_map, origin='lower', cmap='gray_r',
               extent=[0, grid_size, 0, grid_size], vmin=0, vmax=1)
    trail_arr = np.array(trail)
    plt.plot(trail_arr[:, 0], trail_arr[:, 1], 'b-', linewidth=1, label='Robot path')
    if ekf.num_landmarks > 0:
        lm = ekf.landmarks
        plt.scatter(lm[:, 0], lm[:, 1], c='green', marker='^', s=50, label='Landmarks')
    plt.title(f"SLAM Result — {step} scans, {ekf.num_landmarks} landmarks")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.savefig("results/slam_real_map.png", dpi=150, bbox_inches='tight')
    print(f"\n  Map saved: results/slam_real_map.png")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time SLAM with CSPC LiDAR sensor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_real.py --port COM3 --version 2
  python main_real.py --port COM3 --version 2 --baud 230400
  python main_real.py --port /dev/ttyUSB0 --version 3 --max-range 5.0
  python main_real.py --port COM3 --version 2 --grid-size 10
        """
    )
    parser.add_argument("--port", default="COM3",
                        help="Serial port (default: COM3)")
    parser.add_argument("--version", type=int, default=2, choices=[1, 2, 3, 4],
                        help="LiDAR model version (default: 2)")
    parser.add_argument("--baud", type=int, default=None,
                        help="Baud rate (default: auto from version)")
    parser.add_argument("--max-range", type=float, default=8.0,
                        help="Max range in meters (default: 8.0)")
    parser.add_argument("--min-range", type=float, default=0.10,
                        help="Min range in meters (default: 0.10)")
    parser.add_argument("--grid-size", type=float, default=20.0,
                        help="Map size in meters (default: 20.0)")
    parser.add_argument("--grid-resolution", type=float, default=0.1,
                        help="Grid resolution m/cell (default: 0.1)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max scans to process, 0=unlimited (default: 0)")

    args = parser.parse_args()

    run_real_slam(
        port=args.port,
        version=args.version,
        baud_rate=args.baud,
        max_range=args.max_range,
        min_range=args.min_range,
        grid_size=args.grid_size,
        grid_resolution=args.grid_resolution,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
