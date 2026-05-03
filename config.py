"""
Global configuration for the simplified SLAM system.
Parameters for environment, robot, sensors, and algorithms.
"""

import numpy as np

# === Environment ===
ENV_WIDTH = 20.0          # meters
ENV_HEIGHT = 20.0         # meters
WALL_THICKNESS = 0.1      # meters

# === Robot ===
ROBOT_RADIUS = 0.3        # meters
ROBOT_MAX_SPEED = 1.0     # m/s
ROBOT_MAX_OMEGA = np.pi/2 # rad/s (maximum angular velocity)
DT = 0.1                  # time step (seconds)

# === Motion Noise ===
MOTION_NOISE_V = 0.02     # linear velocity standard deviation (m/s)
MOTION_NOISE_W = 0.01     # angular velocity standard deviation (rad/s)

# === LiDAR ===
LIDAR_MAX_RANGE = 8.0     # meters
LIDAR_MIN_RANGE = 0.1     # meters
LIDAR_NUM_BEAMS = 180     # number of beams
LIDAR_FOV = 2 * np.pi     # field of view (360°)
LIDAR_NOISE_STD = 0.05    # noise standard deviation (meters)
LIDAR_MISS_PROB = 0.02    # missed reading probability

# === EKF-SLAM ===
EKF_INITIAL_LANDMARK_COV = 1.0   # initial landmark covariance
EKF_RANGE_NOISE = 0.1            # range measurement noise
EKF_BEARING_NOISE = 0.05         # bearing measurement noise
EKF_ASSOCIATION_THRESHOLD = 1.5  # Mahalanobis threshold for association

# === Occupancy Grid ===
GRID_RESOLUTION = 0.1     # meters per cell
GRID_LOG_ODD_FREE = -0.4  # log-odds for free space
GRID_LOG_ODD_OCC = 0.9    # log-odds for occupied cell
GRID_LOG_ODD_PRIOR = 0.0  # log-odds prior
GRID_LOG_ODD_MAX = 5.0    # maximum saturation
GRID_LOG_ODD_MIN = -5.0   # minimum saturation

# === Visualization ===
VIS_FPS = 10              # frames per second
VIS_TRAIL_LENGTH = 200    # robot trail length
VIS_FIGSIZE = (14, 6)     # figure size

# === Simulation ===
SIM_STEPS = 500           # total simulation steps
SIM_SEED = 42             # seed for reproducibility
