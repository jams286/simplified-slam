"""
EKF-SLAM (Extended Kalman Filter SLAM) implementation.
Simultaneously estimates the robot pose and landmark positions.

Augmented state: μ = [x, y, θ, lx₁, ly₁, lx₂, ly₂, ...]
Covariance: Σ ∈ ℝ^(3+2N)×(3+2N)
"""

import numpy as np
from typing import List, Tuple, Optional


class EKFSLAM:
    """
    EKF-SLAM with point landmarks.
    
    The state contains the robot pose and the positions of all
    observed landmarks. The covariance captures the joint
    pose-landmarks uncertainty.
    
    Observation model:
        z = [r, φ] = [√((lx-x)²+(ly-y)²), atan2(ly-y, lx-x) - θ]
    """

    def __init__(self, initial_pose: np.ndarray,
                 range_noise: float = 0.1,
                 bearing_noise: float = 0.05,
                 association_threshold: float = 1.5):
        # State: [x, y, θ, lx1, ly1, lx2, ly2, ...]
        self.mu = initial_pose.copy().astype(float)
        # Initial covariance (very low for pose)
        self.sigma = np.diag([0.01, 0.01, 0.001])
        
        self.range_noise = range_noise
        self.bearing_noise = bearing_noise
        self.association_threshold = association_threshold
        
        self.num_landmarks = 0
        self.landmark_seen_count: List[int] = []
        
        # Measurement covariance
        self.Q_obs = np.diag([range_noise**2, bearing_noise**2])
        
        # History of estimated poses
        self.estimated_path: List[Tuple[float, float, float]] = [
            tuple(initial_pose)
        ]

    @property
    def robot_pose(self) -> np.ndarray:
        """Current estimated robot pose [x, y, θ]."""
        return self.mu[:3].copy()

    @property
    def landmarks(self) -> np.ndarray:
        """Estimated landmark positions as array (N, 2)."""
        if self.num_landmarks == 0:
            return np.empty((0, 2))
        lm = self.mu[3:].reshape(-1, 2)
        return lm

    def predict(self, v: float, omega: float, dt: float,
                motion_noise: Optional[np.ndarray] = None):
        """
        Prediction step: propagates the state using the motion model.
        
        μ̄ = g(μ, u)
        Σ̄ = Fₓ·Σ·Fₓᵀ + Fᵤ·R·Fᵤᵀ
        
        Args:
            v: linear velocity
            omega: angular velocity  
            dt: time step
            motion_noise: process noise covariance (3x3)
        """
        n = len(self.mu)
        x, y, theta = self.mu[:3]
        
        # State prediction (only pose, landmarks don't change)
        self.mu[0] += v * np.cos(theta) * dt
        self.mu[1] += v * np.sin(theta) * dt
        self.mu[2] += omega * dt
        self.mu[2] = self._normalize_angle(self.mu[2])
        
        # Jacobian of the motion model (with respect to pose)
        G = np.eye(n)
        G[0, 2] = -v * np.sin(theta) * dt
        G[1, 2] = v * np.cos(theta) * dt
        
        # Process noise
        if motion_noise is None:
            R = np.diag([
                (0.1 * abs(v) + 0.01)**2 * dt,
                (0.1 * abs(v) + 0.01)**2 * dt,
                (0.1 * abs(omega) + 0.01)**2 * dt
            ])
        else:
            R = motion_noise
        
        # Update covariance
        self.sigma = G @ self.sigma @ G.T
        self.sigma[:3, :3] += R

    def update(self, observations: List[Tuple[float, float]]):
        """
        Correction step: incorporates landmark observations.
        
        For each observation z = [range, angle]:
        1. Associate with existing landmark or create new
        2. Compute innovation and Jacobian
        3. Update state and covariance
        
        Args:
            observations: list of (range, global_angle) of detected landmarks
        """
        for z_range, z_bearing in observations:
            # Convert bearing to robot-relative
            z_bearing_rel = self._normalize_angle(z_bearing - self.mu[2])
            z = np.array([z_range, z_bearing_rel])
            
            # Expected landmark position in global coordinates
            lx_obs = self.mu[0] + z_range * np.cos(z_bearing)
            ly_obs = self.mu[1] + z_range * np.sin(z_bearing)
            
            # Data association
            landmark_idx = self._data_association(z_range, z_bearing)
            
            if landmark_idx is None:
                # New landmark
                self._add_landmark(lx_obs, ly_obs)
            else:
                # Update with existing landmark
                self._update_landmark(landmark_idx, z)

    def _data_association(self, z_range: float, 
                          z_bearing: float) -> Optional[int]:
        """
        Associates an observation with an existing landmark using
        Mahalanobis distance.
        
        Returns:
            Index of the associated landmark, or None if it's new
        """
        if self.num_landmarks == 0:
            return None
        
        x, y, theta = self.mu[:3]
        # Observed position in global frame
        lx_obs = x + z_range * np.cos(z_bearing)
        ly_obs = y + z_range * np.sin(z_bearing)
        
        min_dist = float('inf')
        best_idx = None
        
        for i in range(self.num_landmarks):
            lx = self.mu[3 + 2*i]
            ly = self.mu[3 + 2*i + 1]
            
            # Simple Euclidean distance as first approximation
            dist = np.hypot(lx - lx_obs, ly - ly_obs)
            
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        if min_dist < self.association_threshold:
            return best_idx
        return None

    def _add_landmark(self, lx: float, ly: float):
        """Adds a new landmark to the state."""
        # Extend state
        self.mu = np.append(self.mu, [lx, ly])
        
        # Extend covariance
        n = len(self.sigma)
        new_sigma = np.zeros((n + 2, n + 2))
        new_sigma[:n, :n] = self.sigma
        # High initial uncertainty for the new landmark
        new_sigma[n, n] = 1.0
        new_sigma[n+1, n+1] = 1.0
        self.sigma = new_sigma
        
        self.num_landmarks += 1
        self.landmark_seen_count.append(1)

    def _update_landmark(self, idx: int, z: np.ndarray):
        """
        EKF update for an observed landmark.
        
        Innovation: ν = z - ẑ
        Jacobian: H = ∂h/∂[x, y, θ, lx, ly]
        Gain: K = Σ·Hᵀ·(H·Σ·Hᵀ + Q)⁻¹
        
        μ = μ + K·ν
        Σ = (I - K·H)·Σ
        """
        n = len(self.mu)
        x, y, theta = self.mu[:3]
        lx = self.mu[3 + 2*idx]
        ly = self.mu[3 + 2*idx + 1]
        
        # Expected measurement
        dx = lx - x
        dy = ly - y
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)
        
        if sqrt_q < 1e-6:
            return  # Avoid division by zero
        
        z_hat = np.array([
            sqrt_q,
            self._normalize_angle(np.arctan2(dy, dx) - theta)
        ])
        
        # Innovation
        innovation = z - z_hat
        innovation[1] = self._normalize_angle(innovation[1])
        
        # Jacobian H (2 x n)
        H = np.zeros((2, n))
        # Derivatives with respect to robot pose
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, 2] = 0
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1
        
        # Derivatives with respect to landmark
        li = 3 + 2 * idx
        H[0, li] = dx / sqrt_q
        H[0, li+1] = dy / sqrt_q
        H[1, li] = -dy / q
        H[1, li+1] = dx / q
        
        # Kalman gain
        S = H @ self.sigma @ H.T + self.Q_obs
        try:
            K = self.sigma @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # Singular matrix, skip update
        
        # Update state and covariance
        self.mu += K @ innovation
        self.mu[2] = self._normalize_angle(self.mu[2])
        
        I_KH = np.eye(n) - K @ H
        self.sigma = I_KH @ self.sigma
        # Symmetrize for numerical stability
        self.sigma = (self.sigma + self.sigma.T) / 2
        
        self.landmark_seen_count[idx] += 1

    def get_pose_covariance(self) -> np.ndarray:
        """Returns the pose covariance sub-matrix (3x3)."""
        return self.sigma[:3, :3].copy()

    def get_landmark_covariance(self, idx: int) -> np.ndarray:
        """Returns the covariance of a specific landmark (2x2)."""
        li = 3 + 2 * idx
        return self.sigma[li:li+2, li:li+2].copy()

    def step(self, v: float, omega: float, dt: float,
             observations: List[Tuple[float, float]]):
        """
        Executes a complete predict-update cycle.
        
        Args:
            v, omega: motion controls
            dt: time step
            observations: list of (range, angle) of landmarks
        """
        self.predict(v, omega, dt)
        if observations:
            self.update(observations)
        self.estimated_path.append(tuple(self.mu[:3]))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalizes angle to the range [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
