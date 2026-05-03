"""
2D environment simulator with walls and obstacles.
Defines the world where the robot navigates and the elements detected by the LiDAR.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Segment:
    """Line segment representing a wall or obstacle."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Landmark:
    """Reference point in the environment (corners, notable points)."""
    x: float
    y: float
    landmark_id: int


class Environment:
    """
    2D rectangular environment with perimeter walls and internal obstacles.
    
    The environment is composed of line segments that represent
    reflective surfaces for the LiDAR sensor.
    """

    def __init__(self, width: float = 20.0, height: float = 20.0):
        self.width = width
        self.height = height
        self.walls: List[Segment] = []
        self.landmarks: List[Landmark] = []
        self._build_boundary()

    def _build_boundary(self):
        """Builds the perimeter walls of the environment."""
        w, h = self.width, self.height
        self.walls.extend([
            Segment(0, 0, w, 0),     # bottom wall
            Segment(w, 0, w, h),     # right wall
            Segment(w, h, 0, h),     # top wall
            Segment(0, h, 0, 0),     # left wall
        ])

    def add_rectangle(self, cx: float, cy: float, rw: float, rh: float):
        """
        Adds a rectangular obstacle.
        
        Args:
            cx, cy: rectangle center
            rw, rh: rectangle width and height
        """
        x1, y1 = cx - rw / 2, cy - rh / 2
        x2, y2 = cx + rw / 2, cy + rh / 2
        self.walls.extend([
            Segment(x1, y1, x2, y1),
            Segment(x2, y1, x2, y2),
            Segment(x2, y2, x1, y2),
            Segment(x1, y2, x1, y1),
        ])
        # Add corners as landmarks
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for x, y in corners:
            lid = len(self.landmarks)
            self.landmarks.append(Landmark(x, y, lid))

    def add_segment(self, x1: float, y1: float, x2: float, y2: float):
        """Adds an individual wall segment."""
        self.walls.append(Segment(x1, y1, x2, y2))
        # Endpoints as landmarks
        for x, y in [(x1, y1), (x2, y2)]:
            lid = len(self.landmarks)
            self.landmarks.append(Landmark(x, y, lid))

    def ray_cast(self, ox: float, oy: float, angle: float, 
                 max_range: float) -> float:
        """
        Casts a ray from (ox, oy) in direction 'angle' and returns
        the distance to the first obstacle, or max_range if no intersection.
        
        Uses parameterized ray-segment intersection.
        """
        dx = np.cos(angle)
        dy = np.sin(angle)
        min_dist = max_range

        for wall in self.walls:
            dist = self._ray_segment_intersection(
                ox, oy, dx, dy,
                wall.x1, wall.y1, wall.x2, wall.y2
            )
            if dist is not None and dist < min_dist:
                min_dist = dist

        return min_dist

    @staticmethod
    def _ray_segment_intersection(ox: float, oy: float, dx: float, dy: float,
                                   x1: float, y1: float, x2: float, y2: float):
        """
        Computes the intersection between a ray and a segment using
        the parameterized method (Cramer).
        
        Returns the distance t if there is a valid intersection, None otherwise.
        """
        sx = x2 - x1
        sy = y2 - y1

        denom = dx * sy - dy * sx
        if abs(denom) < 1e-10:
            return None  # Ray parallel to segment

        t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom

        if t >= 0 and 0 <= u <= 1:
            return t
        return None

    def is_free(self, x: float, y: float, radius: float = 0.0) -> bool:
        """Checks if a position is valid (no collision)."""
        if x - radius < 0 or x + radius > self.width:
            return False
        if y - radius < 0 or y + radius > self.height:
            return False
        # Check distance to each segment
        for wall in self.walls[4:]:  # Skip perimeter walls
            dist = self._point_segment_distance(
                x, y, wall.x1, wall.y1, wall.x2, wall.y2
            )
            if dist < radius:
                return False
        return True

    @staticmethod
    def _point_segment_distance(px: float, py: float,
                                 x1: float, y1: float,
                                 x2: float, y2: float) -> float:
        """Minimum distance from a point to a segment."""
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-10:
            return np.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return np.hypot(px - proj_x, py - proj_y)

    def get_ground_truth_map(self, resolution: float = 0.1) -> np.ndarray:
        """
        Generates a ground truth map as a binary grid.
        
        Returns:
            Matrix where 1 = occupied, 0 = free
        """
        rows = int(self.height / resolution)
        cols = int(self.width / resolution)
        grid = np.zeros((rows, cols), dtype=np.float32)

        for wall in self.walls:
            # Rasterize each segment into the grid
            n_samples = int(np.hypot(wall.x2 - wall.x1, wall.y2 - wall.y1) / resolution * 2) + 1
            for i in range(n_samples):
                t = i / max(n_samples - 1, 1)
                wx = wall.x1 + t * (wall.x2 - wall.x1)
                wy = wall.y1 + t * (wall.y2 - wall.y1)
                col = int(wx / resolution)
                row = int(wy / resolution)
                if 0 <= row < rows and 0 <= col < cols:
                    grid[row, col] = 1.0

        return grid


def create_default_environment() -> Environment:
    """Creates a predefined environment with varied obstacles for demonstration."""
    env = Environment(20.0, 20.0)

    # Rooms and corridors
    env.add_rectangle(5.0, 5.0, 3.0, 3.0)    # bottom-left obstacle
    env.add_rectangle(15.0, 5.0, 2.0, 4.0)   # bottom-right obstacle
    env.add_rectangle(5.0, 15.0, 2.5, 2.5)   # top-left obstacle
    env.add_rectangle(15.0, 15.0, 3.0, 2.0)  # top-right obstacle
    env.add_rectangle(10.0, 10.0, 2.0, 2.0)  # central obstacle

    # Internal walls (corridors)
    env.add_segment(8.0, 0.0, 8.0, 4.0)      # bottom vertical wall
    env.add_segment(12.0, 7.0, 12.0, 13.0)   # central vertical wall
    env.add_segment(0.0, 12.0, 4.0, 12.0)    # left horizontal wall

    return env
