"""
Real LiDAR adapter.
Bridges the CSPC LiDAR driver (drivers/cspc_lidar/cspc_lidar.py) to the
SLAM system's LiDARScan format.
"""

import sys
import os
import numpy as np
import time
from typing import Optional, Generator

# Add the driver path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drivers', 'cspc_lidar'))

from cspc_lidar import CSPCLidar, LidarScan as DriverScan
from .lidar import LiDARScan


class RealLiDAR:
    """
    Wrapper around the CSPC LiDAR driver that produces LiDARScan objects
    compatible with the SLAM system.
    
    Usage:
        lidar = RealLiDAR(port="COM3", version=2)
        lidar.start()
        for scan in lidar.iter_scans():
            # scan is a LiDARScan object ready for SLAM
            ...
        lidar.stop()
    """

    def __init__(self, port: str = "COM3", version: int = 2,
                 baud_rate: Optional[int] = None,
                 max_range: float = 8.0, min_range: float = 0.10,
                 num_beams: int = 360):
        self.port = port
        self.version = version
        self.baud_rate = baud_rate
        self.max_range = max_range
        self.min_range = min_range
        self.num_beams = num_beams  # Target number of beams for uniform scan

        self._driver = CSPCLidar(
            port=port,
            version=version,
            baud_rate=baud_rate,
            min_range=min_range,
            max_range=max_range,
        )
        self._running = False

    def start(self):
        """Connect and start the LiDAR."""
        self._driver.start()
        self._running = True

    def stop(self):
        """Stop the LiDAR and disconnect."""
        self._running = False
        self._driver.stop()

    def iter_scans(self, robot_pose: Optional[np.ndarray] = None) -> Generator[LiDARScan, None, None]:
        """
        Generator that yields LiDARScan objects compatible with the SLAM system.
        
        Args:
            robot_pose: If provided, used as the robot pose in the scan.
                       If None, defaults to (0, 0, 0) — the pose will be
                       set externally by the SLAM loop.
        """
        for driver_scan in self._driver.iter_scans():
            if not self._running:
                break
            slam_scan = self._convert_scan(driver_scan, robot_pose)
            if slam_scan is not None:
                yield slam_scan

    def get_one_scan(self, robot_pose: Optional[np.ndarray] = None,
                     timeout: float = 5.0) -> Optional[LiDARScan]:
        """Get a single scan, blocking."""
        driver_scan = self._driver.get_one_scan(timeout=timeout)
        if driver_scan is None:
            return None
        return self._convert_scan(driver_scan, robot_pose)

    def _convert_scan(self, driver_scan: DriverScan,
                      robot_pose: Optional[np.ndarray] = None) -> Optional[LiDARScan]:
        """
        Convert a driver LidarScan to the SLAM system's LiDARScan format.
        
        The driver provides variable-count points at non-uniform angles.
        We resample into uniform angular bins for consistency with the SLAM algorithms.
        """
        if not driver_scan.points:
            return None

        # Extract raw angles and ranges
        raw_angles = np.array([p.angle_deg for p in driver_scan.points])
        raw_ranges = np.array([p.range_m for p in driver_scan.points])
        raw_quality = np.array([p.quality for p in driver_scan.points])

        # Create uniform angular bins (0 to 360 degrees)
        uniform_angles_deg = np.linspace(0, 360, self.num_beams, endpoint=False)
        uniform_angles_rad = np.deg2rad(uniform_angles_deg)

        # Bin the raw points into uniform angles (nearest-neighbor interpolation)
        ranges = np.full(self.num_beams, self.max_range)
        valid = np.zeros(self.num_beams, dtype=bool)

        for i, angle_deg in enumerate(raw_angles):
            if raw_ranges[i] <= 0 or raw_ranges[i] > self.max_range:
                continue
            if raw_ranges[i] < self.min_range:
                continue

            # Find the closest bin
            bin_idx = int(round(angle_deg / 360.0 * self.num_beams)) % self.num_beams

            # Keep the closest measurement if multiple fall in same bin
            if not valid[bin_idx] or raw_ranges[i] < ranges[bin_idx]:
                ranges[bin_idx] = raw_ranges[i]
                valid[bin_idx] = True

        # Default robot pose at origin if not provided
        if robot_pose is None:
            robot_pose = np.array([0.0, 0.0, 0.0])

        rx, ry, rtheta = robot_pose

        # Angles in global frame (relative to robot heading)
        global_angles = uniform_angles_rad + rtheta

        return LiDARScan(
            ranges=ranges,
            angles=global_angles,
            valid=valid,
            robot_pose=(rx, ry, rtheta),
            max_range=self.max_range
        )

    def scan_to_local_points(self, scan: LiDARScan) -> np.ndarray:
        """
        Convert a scan to points in the LOCAL robot frame (for scan matching).
        This ignores robot_pose and returns points relative to the sensor.
        
        Returns:
            (N, 2) array of [x, y] points in local frame
        """
        valid_ranges = scan.ranges[scan.valid]
        # Use angles without the robot heading offset
        num_valid = np.sum(scan.valid)
        uniform_angles = np.linspace(0, 2 * np.pi, len(scan.ranges), endpoint=False)
        valid_angles = uniform_angles[scan.valid]

        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)

        return np.column_stack([x, y])
