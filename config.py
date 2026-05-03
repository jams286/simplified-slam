"""
Configuración global del sistema SLAM simplificado.
Parámetros del entorno, robot, sensores y algoritmos.
"""

import numpy as np

# === Entorno ===
ENV_WIDTH = 20.0          # metros
ENV_HEIGHT = 20.0         # metros
WALL_THICKNESS = 0.1      # metros

# === Robot ===
ROBOT_RADIUS = 0.3        # metros
ROBOT_MAX_SPEED = 1.0     # m/s
ROBOT_MAX_OMEGA = np.pi/2 # rad/s (velocidad angular máxima)
DT = 0.1                  # paso de tiempo (segundos)

# === Ruido de movimiento ===
MOTION_NOISE_V = 0.02     # desviación estándar velocidad lineal (m/s)
MOTION_NOISE_W = 0.01     # desviación estándar velocidad angular (rad/s)

# === LiDAR ===
LIDAR_MAX_RANGE = 8.0     # metros
LIDAR_MIN_RANGE = 0.1     # metros
LIDAR_NUM_BEAMS = 180     # número de rayos
LIDAR_FOV = 2 * np.pi     # campo de visión (360°)
LIDAR_NOISE_STD = 0.05    # desviación estándar del ruido (metros)
LIDAR_MISS_PROB = 0.02    # probabilidad de lectura perdida

# === EKF-SLAM ===
EKF_INITIAL_LANDMARK_COV = 1.0   # covarianza inicial de landmarks
EKF_RANGE_NOISE = 0.1            # ruido en medición de rango
EKF_BEARING_NOISE = 0.05         # ruido en medición de ángulo
EKF_ASSOCIATION_THRESHOLD = 1.5  # umbral Mahalanobis para asociación

# === Mapa de Ocupación ===
GRID_RESOLUTION = 0.1     # metros por celda
GRID_LOG_ODD_FREE = -0.4  # log-odds para espacio libre
GRID_LOG_ODD_OCC = 0.9    # log-odds para celda ocupada
GRID_LOG_ODD_PRIOR = 0.0  # log-odds prior
GRID_LOG_ODD_MAX = 5.0    # saturación máxima
GRID_LOG_ODD_MIN = -5.0   # saturación mínima

# === Visualización ===
VIS_FPS = 10              # cuadros por segundo
VIS_TRAIL_LENGTH = 200    # longitud del rastro del robot
VIS_FIGSIZE = (14, 6)     # tamaño de la figura

# === Simulación ===
SIM_STEPS = 500           # pasos totales de simulación
SIM_SEED = 42             # semilla para reproducibilidad
