"""
Scan Matching using ICP (Iterative Closest Point).
Estimates the rigid-body transformation between consecutive LiDAR scans
to provide odometry when no wheel encoders are available (handheld mode).
"""

import numpy as np
from typing import Tuple, Optional


class ScanMatcher:
    """
    ICP-based scan matcher for estimating motion between consecutive scans.
    
    Algorithm:
        1. For each point in the new scan, find the closest point in the reference scan
        2. Compute the optimal rotation + translation (SVD method)
        3. Apply transformation and repeat until convergence
    
    Returns (dx, dy, dtheta) — the estimated motion between scans.
    """

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-4,
                 max_correspondence_dist: float = 1.0,
                 min_points: int = 20):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_dist = max_correspondence_dist
        self.min_points = min_points

    def match(self, reference: np.ndarray, current: np.ndarray,
              initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Estimate the transformation from reference frame to current frame.
        
        Args:
            reference: (N, 2) array of points from the previous scan
            current: (M, 2) array of points from the new scan
            initial_guess: [dx, dy, dtheta] initial estimate (default: zeros)
            
        Returns:
            transform: [dx, dy, dtheta] — estimated motion
            error: mean squared correspondence error (quality indicator)
        """
        if len(reference) < self.min_points or len(current) < self.min_points:
            return np.zeros(3), float('inf')

        # Apply initial guess
        if initial_guess is not None:
            T = self._pose_to_matrix(initial_guess)
            src = self._transform_points(current, T)
        else:
            src = current.copy()

        cumulative_T = self._pose_to_matrix(initial_guess) if initial_guess is not None else np.eye(3)
        prev_error = float('inf')

        for iteration in range(self.max_iterations):
            # Step 1: Find closest correspondences
            indices, distances = self._find_correspondences(reference, src)

            # Filter by max distance
            valid = distances < self.max_correspondence_dist
            if np.sum(valid) < self.min_points:
                break

            matched_ref = reference[indices[valid]]
            matched_src = src[valid]

            # Step 2: Compute optimal transformation (SVD)
            R, t = self._compute_transform_svd(matched_src, matched_ref)

            # Step 3: Apply transformation
            T_step = np.eye(3)
            T_step[:2, :2] = R
            T_step[:2, 2] = t
            src = self._transform_points(src, T_step)
            cumulative_T = T_step @ cumulative_T

            # Check convergence
            mean_error = np.mean(distances[valid])
            if abs(prev_error - mean_error) < self.tolerance:
                break
            prev_error = mean_error

        # Extract pose from cumulative transformation
        dx = cumulative_T[0, 2]
        dy = cumulative_T[1, 2]
        dtheta = np.arctan2(cumulative_T[1, 0], cumulative_T[0, 0])

        return np.array([dx, dy, dtheta]), prev_error

    def _find_correspondences(self, reference: np.ndarray, 
                               source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest neighbor for each source point in reference."""
        # Brute-force nearest neighbor (efficient enough for LiDAR point counts)
        indices = np.zeros(len(source), dtype=int)
        distances = np.zeros(len(source))

        for i, pt in enumerate(source):
            dists = np.sum((reference - pt) ** 2, axis=1)
            indices[i] = np.argmin(dists)
            distances[i] = np.sqrt(dists[indices[i]])

        return indices, distances

    def _compute_transform_svd(self, source: np.ndarray, 
                                target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal rotation and translation using SVD.
        Minimizes sum of squared distances between matched points.
        """
        # Centroids
        centroid_src = np.mean(source, axis=0)
        centroid_tgt = np.mean(target, axis=0)

        # Center the points
        src_centered = source - centroid_src
        tgt_centered = target - centroid_tgt

        # Cross-covariance matrix
        H = src_centered.T @ tgt_centered

        # SVD
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Translation
        t = centroid_tgt - R @ centroid_src

        return R, t

    def _pose_to_matrix(self, pose: np.ndarray) -> np.ndarray:
        """Convert [dx, dy, dtheta] to 3x3 homogeneous transformation matrix."""
        dx, dy, dtheta = pose
        c, s = np.cos(dtheta), np.sin(dtheta)
        T = np.array([
            [c, -s, dx],
            [s,  c, dy],
            [0,  0,  1]
        ])
        return T

    def _transform_points(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Apply homogeneous transformation to 2D points."""
        ones = np.ones((len(points), 1))
        homogeneous = np.hstack([points, ones])
        transformed = (T @ homogeneous.T).T
        return transformed[:, :2]
