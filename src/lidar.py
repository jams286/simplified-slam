"""
2D LiDAR sensor simulator with realistic Gaussian noise.
Generates simulated scans via ray-casting in the environment.
"""

import numpy as np
from typing import Tuple, List, Optional
from .environment import Environment


class LiDAR:
    """
    Simulated 2D LiDAR sensor with realistic noise model.
    
    Model characteristics:
    - Gaussian noise on distance measurements
    - Probability of missed readings (max_range)
    - Configurable angular resolution
    - Adjustable field of view
    """

    def __init__(self, num_beams: int = 180, max_range: float = 8.0,
                 min_range: float = 0.1, fov: float = 2 * np.pi,
                 noise_std: float = 0.05, miss_probability: float = 0.02):
        self.num_beams = num_beams
        self.max_range = max_range
        self.min_range = min_range
        self.fov = fov
        self.noise_std = noise_std
        self.miss_probability = miss_probability
        
        # Angles of beams relative to robot orientation
        self.angles = np.linspace(-fov / 2, fov / 2, num_beams, endpoint=False)

    def scan(self, environment: Environment, robot_x: float, 
             robot_y: float, robot_theta: float) -> 'LiDARScan':
        """
        Performs a complete scan of the environment from the robot's pose.
        
        Args:
            environment: environment with obstacles
            robot_x, robot_y: robot position
            robot_theta: robot orientation
            
        Returns:
            LiDARScan object with the measurements
        """
        ranges = np.zeros(self.num_beams)
        valid = np.ones(self.num_beams, dtype=bool)

        for i, local_angle in enumerate(self.angles):
            global_angle = robot_theta + local_angle
            
            # Noise-free ray-casting
            true_range = environment.ray_cast(
                robot_x, robot_y, global_angle, self.max_range
            )
            
            # Gaussian noise model
            noisy_range = true_range + np.random.normal(0, self.noise_std)
            
            # Simulate missed readings
            if np.random.random() < self.miss_probability:
                noisy_range = self.max_range
                valid[i] = False
            
            # Clamp to valid range
            noisy_range = np.clip(noisy_range, self.min_range, self.max_range)
            
            # Mark as invalid if at the limit
            if noisy_range >= self.max_range - 0.01:
                valid[i] = False
            
            ranges[i] = noisy_range

        return LiDARScan(
            ranges=ranges,
            angles=self.angles + robot_theta,
            valid=valid,
            robot_pose=(robot_x, robot_y, robot_theta),
            max_range=self.max_range
        )

    def extract_landmarks(self, scan: 'LiDARScan', 
                          min_cluster_size: int = 3,
                          max_cluster_gap: float = 0.3) -> List[Tuple[float, float]]:
        """
        Extracts landmarks (notable points) from the scan using
        discontinuity detection and clustering.
        
        Detects corners/points where there are abrupt distance jumps.
        
        Returns:
            List of (range, angle) of detected landmarks
        """
        landmarks = []
        ranges = scan.ranges
        valid = scan.valid
        
        # Detect discontinuities (possible corners)
        for i in range(1, len(ranges) - 1):
            if not valid[i]:
                continue
            
            # Difference with neighbors
            if valid[i-1] and valid[i+1]:
                diff_left = abs(ranges[i] - ranges[i-1])
                diff_right = abs(ranges[i] - ranges[i+1])
                
                # Discontinuity point
                if diff_left > max_cluster_gap or diff_right > max_cluster_gap:
                    landmarks.append((ranges[i], scan.angles[i]))
        
        return landmarks

    def scan_to_points(self, scan: 'LiDARScan') -> np.ndarray:
        """
        Converts a scan to Cartesian points in the global frame.
        
        Returns:
            Array (N, 2) with [x, y] coordinates of valid points
        """
        valid_ranges = scan.ranges[scan.valid]
        valid_angles = scan.angles[scan.valid]
        
        rx, ry, _ = scan.robot_pose
        
        x = rx + valid_ranges * np.cos(valid_angles)
        y = ry + valid_ranges * np.sin(valid_angles)
        
        return np.column_stack([x, y])


class LiDARScan:
    """Result of a LiDAR scan."""
    
    def __init__(self, ranges: np.ndarray, angles: np.ndarray,
                 valid: np.ndarray, robot_pose: Tuple[float, float, float],
                 max_range: float):
        self.ranges = ranges
        self.angles = angles
        self.valid = valid
        self.robot_pose = robot_pose
        self.max_range = max_range

    @property
    def num_valid(self) -> int:
        """Number of valid measurements."""
        return int(np.sum(self.valid))

    def get_endpoints(self) -> np.ndarray:
        """
        Computes the ray endpoints in global coordinates.
        
        Returns:
            Array (N, 2) with [x, y] coordinates of all endpoints
        """
        rx, ry, _ = self.robot_pose
        x = rx + self.ranges * np.cos(self.angles)
        y = ry + self.ranges * np.sin(self.angles)
        return np.column_stack([x, y])

    def get_valid_endpoints(self) -> np.ndarray:
        """Returns only the endpoints of valid measurements."""
        endpoints = self.get_endpoints()
        return endpoints[self.valid]
