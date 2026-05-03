"""
Differential drive robot kinematic model.
Manages the state (x, y, θ) and the motion model with noise.
"""

import numpy as np
from typing import Tuple


class Robot:
    """
    2D differential drive robot with velocity-based motion model.
    
    State: [x, y, θ] where θ is the orientation in radians.
    Control: [v, ω] linear and angular velocity.
    """

    def __init__(self, x: float = 10.0, y: float = 10.0, theta: float = 0.0,
                 noise_v: float = 0.02, noise_w: float = 0.01):
        self.x = x
        self.y = y
        self.theta = theta
        self.noise_v = noise_v
        self.noise_w = noise_w
        # History of real poses (ground truth)
        self.true_path: list = [(x, y, theta)]

    @property
    def pose(self) -> np.ndarray:
        """Returns the current state as vector [x, y, θ]."""
        return np.array([self.x, self.y, self.theta])

    def move(self, v: float, omega: float, dt: float,
             add_noise: bool = True) -> np.ndarray:
        """
        Executes a motion step using the velocity motion model.
        
        Kinematic model:
            x' = x + v·cos(θ)·dt
            y' = y + v·sin(θ)·dt
            θ' = θ + ω·dt
        
        Args:
            v: linear velocity (m/s)
            omega: angular velocity (rad/s)
            dt: time step (s)
            add_noise: whether to add process noise
            
        Returns:
            New state [x, y, θ]
        """
        if add_noise:
            v_noisy = v + np.random.normal(0, self.noise_v)
            omega_noisy = omega + np.random.normal(0, self.noise_w)
        else:
            v_noisy = v
            omega_noisy = omega

        self.x += v_noisy * np.cos(self.theta) * dt
        self.y += v_noisy * np.sin(self.theta) * dt
        self.theta += omega_noisy * dt
        self.theta = self._normalize_angle(self.theta)

        self.true_path.append((self.x, self.y, self.theta))
        return self.pose

    @staticmethod
    def motion_model(state: np.ndarray, v: float, omega: float, 
                     dt: float) -> np.ndarray:
        """
        Noise-free motion model (for EKF prediction).
        
        Args:
            state: [x, y, θ]
            v, omega: controls
            dt: time step
            
        Returns:
            Predicted state [x', y', θ']
        """
        x, y, theta = state
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + omega * dt
        theta_new = Robot._normalize_angle(theta_new)
        return np.array([x_new, y_new, theta_new])

    @staticmethod
    def motion_jacobian(state: np.ndarray, v: float, omega: float,
                        dt: float) -> np.ndarray:
        """
        Jacobian of the motion model with respect to the state.
        
        F = ∂f/∂x = [[1, 0, -v·sin(θ)·dt],
                      [0, 1,  v·cos(θ)·dt],
                      [0, 0,  1          ]]
        """
        theta = state[2]
        F = np.eye(3)
        F[0, 2] = -v * np.sin(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        return F

    @staticmethod
    def motion_noise_covariance(v: float, omega: float, dt: float,
                                 alpha: np.ndarray = None) -> np.ndarray:
        """
        Process noise covariance Q.
        
        Simplified model proportional to controls.
        """
        if alpha is None:
            alpha = np.array([0.1, 0.01, 0.01, 0.1])
        
        # Noise in control space
        M = np.diag([
            alpha[0] * v**2 + alpha[1] * omega**2,
            alpha[2] * v**2 + alpha[3] * omega**2
        ])
        
        # Jacobian of control-to-state transformation
        theta = 0  # Approximation
        V = np.array([
            [np.cos(theta) * dt, 0],
            [np.sin(theta) * dt, 0],
            [0, dt]
        ])
        
        Q = V @ M @ V.T
        return Q

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalizes angle to the range [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def generate_exploration_commands(num_steps: int, dt: float,
                                   env_width: float = 20.0,
                                   env_height: float = 20.0) -> list:
    """
    Generates a sequence of motion commands to explore the environment.
    Uses a 'lawn mower' exploration pattern with variations.
    
    Returns:
        List of (v, omega) tuples for each step.
    """
    commands = []
    phase_length = int(3.0 / dt)  # 3 seconds per phase
    turn_length = int(1.5 / dt)   # 1.5 seconds to turn
    
    v_cruise = 0.8  # cruise speed
    omega_turn = np.pi / 3  # turn speed
    
    step = 0
    direction = 1  # 1 = right, -1 = left
    
    while step < num_steps:
        # Go straight
        for _ in range(min(phase_length, num_steps - step)):
            commands.append((v_cruise, 0.0))
            step += 1
        
        if step >= num_steps:
            break
            
        # Turn 90°
        for _ in range(min(turn_length, num_steps - step)):
            commands.append((0.2, direction * omega_turn))
            step += 1
        
        if step >= num_steps:
            break
            
        # Advance a little
        short_phase = int(1.5 / dt)
        for _ in range(min(short_phase, num_steps - step)):
            commands.append((v_cruise, 0.0))
            step += 1
        
        if step >= num_steps:
            break
            
        # Turn 90° same direction
        for _ in range(min(turn_length, num_steps - step)):
            commands.append((0.2, direction * omega_turn))
            step += 1
        
        direction *= -1  # Alternate direction

    return commands[:num_steps]
