"""
Occupancy Grid Map.
Implements the inverse sensor model with log-odds for
efficient map updating.
"""

import numpy as np
from typing import Tuple
from .lidar import LiDARScan


class OccupancyGrid:
    """
    Occupancy grid map using log-odds representation.
    
    Probabilistic model:
        L(m|z) = L(m|z₁:ₜ₋₁) + L(m|zₜ) - L₀
        
    Where L = log(p/(1-p)) is the log-odds transformation.
    
    Advantages of log-odds:
    - Additive update (efficient)
    - Avoids numerical issues with probabilities near 0 or 1
    - Natural saturation with limits
    """

    def __init__(self, width: float, height: float, resolution: float = 0.1,
                 log_odd_free: float = -0.4, log_odd_occ: float = 0.9,
                 log_odd_prior: float = 0.0, log_odd_max: float = 5.0,
                 log_odd_min: float = -5.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.log_odd_free = log_odd_free
        self.log_odd_occ = log_odd_occ
        self.log_odd_prior = log_odd_prior
        self.log_odd_max = log_odd_max
        self.log_odd_min = log_odd_min
        
        # Grid dimensions
        self.cols = int(width / resolution)
        self.rows = int(height / resolution)
        
        # Grid in log-odds (initialized with prior)
        self.grid = np.full((self.rows, self.cols), log_odd_prior, 
                           dtype=np.float32)
        
        # Update counter per cell
        self.update_count = np.zeros((self.rows, self.cols), dtype=np.int32)

    def update(self, scan: LiDARScan):
        """
        Updates the map with a LiDAR scan using the
        inverse sensor model.
        
        For each ray:
        - Traversed cells → increment free log-odds
        - Final cell (hit) → increment occupied log-odds
        
        Uses Bresenham to trace rays efficiently.
        """
        rx, ry, _ = scan.robot_pose
        robot_col = int(rx / self.resolution)
        robot_row = int(ry / self.resolution)
        
        endpoints = scan.get_endpoints()
        
        for i in range(len(scan.ranges)):
            ex, ey = endpoints[i]
            end_col = int(ex / self.resolution)
            end_row = int(ey / self.resolution)
            
            # Trace ray with Bresenham
            cells = self._bresenham(robot_col, robot_row, end_col, end_row)
            
            # Free cells (all except the last)
            for col, row in cells[:-1]:
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    self.grid[row, col] += self.log_odd_free
                    self.grid[row, col] = max(self.grid[row, col], 
                                             self.log_odd_min)
                    self.update_count[row, col] += 1
            
            # Occupied cell (last cell, only if the ray hit)
            if scan.valid[i] and cells:
                col, row = cells[-1]
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    self.grid[row, col] += self.log_odd_occ
                    self.grid[row, col] = min(self.grid[row, col], 
                                             self.log_odd_max)
                    self.update_count[row, col] += 1

    def update_from_pose(self, scan: LiDARScan, estimated_pose: np.ndarray):
        """
        Updates the map using the SLAM-estimated pose instead
        of the real robot pose.
        
        Args:
            scan: original LiDAR scan
            estimated_pose: [x, y, θ] estimated by EKF-SLAM
        """
        rx, ry, rtheta = estimated_pose
        robot_col = int(rx / self.resolution)
        robot_row = int(ry / self.resolution)
        
        for i in range(len(scan.ranges)):
            if not scan.valid[i]:
                continue
                
            # Recalculate endpoint with estimated pose
            angle = scan.angles[i] - scan.robot_pose[2] + rtheta
            ex = rx + scan.ranges[i] * np.cos(angle)
            ey = ry + scan.ranges[i] * np.sin(angle)
            
            end_col = int(ex / self.resolution)
            end_row = int(ey / self.resolution)
            
            # Trace ray
            cells = self._bresenham(robot_col, robot_row, end_col, end_row)
            
            for col, row in cells[:-1]:
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    self.grid[row, col] += self.log_odd_free
                    self.grid[row, col] = max(self.grid[row, col], 
                                             self.log_odd_min)
                    self.update_count[row, col] += 1
            
            if cells:
                col, row = cells[-1]
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    self.grid[row, col] += self.log_odd_occ
                    self.grid[row, col] = min(self.grid[row, col], 
                                             self.log_odd_max)
                    self.update_count[row, col] += 1

    def get_probability_map(self) -> np.ndarray:
        """
        Converts log-odds to occupancy probabilities [0, 1].
        
        p = 1 - 1/(1 + exp(L))
        """
        return 1.0 - 1.0 / (1.0 + np.exp(self.grid))

    def get_binary_map(self, threshold: float = 0.6) -> np.ndarray:
        """
        Generates binary map: 1 = occupied, 0 = free.
        
        Args:
            threshold: probability threshold to consider occupied
        """
        prob_map = self.get_probability_map()
        return (prob_map > threshold).astype(np.float32)

    def get_explored_mask(self) -> np.ndarray:
        """Returns mask of cells that have been observed at least once."""
        return self.update_count > 0

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list:
        """
        Bresenham's algorithm to trace a line on the grid.
        
        Returns:
            List of (col, row) tuples of traversed cells.
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return cells

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Converts world coordinates to grid indices."""
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Converts grid indices to world coordinates."""
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return x, y
