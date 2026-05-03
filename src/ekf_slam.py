"""
Implementación de EKF-SLAM (Extended Kalman Filter SLAM).
Estima simultáneamente la pose del robot y la posición de landmarks.

Estado aumentado: μ = [x, y, θ, lx₁, ly₁, lx₂, ly₂, ...]
Covarianza: Σ ∈ ℝ^(3+2N)×(3+2N)
"""

import numpy as np
from typing import List, Tuple, Optional


class EKFSLAM:
    """
    EKF-SLAM con landmarks puntuales.
    
    El estado contiene la pose del robot y las posiciones de todos
    los landmarks observados. La covarianza captura la incertidumbre
    conjunta pose-landmarks.
    
    Modelo de observación:
        z = [r, φ] = [√((lx-x)²+(ly-y)²), atan2(ly-y, lx-x) - θ]
    """

    def __init__(self, initial_pose: np.ndarray,
                 range_noise: float = 0.1,
                 bearing_noise: float = 0.05,
                 association_threshold: float = 1.5):
        # Estado: [x, y, θ, lx1, ly1, lx2, ly2, ...]
        self.mu = initial_pose.copy().astype(float)
        # Covarianza inicial (muy baja para la pose)
        self.sigma = np.diag([0.01, 0.01, 0.001])
        
        self.range_noise = range_noise
        self.bearing_noise = bearing_noise
        self.association_threshold = association_threshold
        
        self.num_landmarks = 0
        self.landmark_seen_count: List[int] = []
        
        # Covarianza de medición
        self.Q_obs = np.diag([range_noise**2, bearing_noise**2])
        
        # Historial de poses estimadas
        self.estimated_path: List[Tuple[float, float, float]] = [
            tuple(initial_pose)
        ]

    @property
    def robot_pose(self) -> np.ndarray:
        """Pose estimada actual del robot [x, y, θ]."""
        return self.mu[:3].copy()

    @property
    def landmarks(self) -> np.ndarray:
        """Posiciones estimadas de landmarks como array (N, 2)."""
        if self.num_landmarks == 0:
            return np.empty((0, 2))
        lm = self.mu[3:].reshape(-1, 2)
        return lm

    def predict(self, v: float, omega: float, dt: float,
                motion_noise: Optional[np.ndarray] = None):
        """
        Paso de predicción: propaga el estado usando el modelo de movimiento.
        
        μ̄ = g(μ, u)
        Σ̄ = Fₓ·Σ·Fₓᵀ + Fᵤ·R·Fᵤᵀ
        
        Args:
            v: velocidad lineal
            omega: velocidad angular  
            dt: paso de tiempo
            motion_noise: covarianza de ruido de proceso (3x3)
        """
        n = len(self.mu)
        x, y, theta = self.mu[:3]
        
        # Predicción del estado (solo pose, landmarks no cambian)
        self.mu[0] += v * np.cos(theta) * dt
        self.mu[1] += v * np.sin(theta) * dt
        self.mu[2] += omega * dt
        self.mu[2] = self._normalize_angle(self.mu[2])
        
        # Jacobiano del modelo de movimiento (respecto a pose)
        G = np.eye(n)
        G[0, 2] = -v * np.sin(theta) * dt
        G[1, 2] = v * np.cos(theta) * dt
        
        # Ruido de proceso
        if motion_noise is None:
            R = np.diag([
                (0.1 * abs(v) + 0.01)**2 * dt,
                (0.1 * abs(v) + 0.01)**2 * dt,
                (0.1 * abs(omega) + 0.01)**2 * dt
            ])
        else:
            R = motion_noise
        
        # Actualizar covarianza
        self.sigma = G @ self.sigma @ G.T
        self.sigma[:3, :3] += R

    def update(self, observations: List[Tuple[float, float]]):
        """
        Paso de corrección: incorpora observaciones de landmarks.
        
        Para cada observación z = [rango, ángulo]:
        1. Asociar con landmark existente o crear nuevo
        2. Calcular innovación y Jacobiano
        3. Actualizar estado y covarianza
        
        Args:
            observations: lista de (rango, ángulo_global) de landmarks detectados
        """
        for z_range, z_bearing in observations:
            # Convertir bearing a relativo al robot
            z_bearing_rel = self._normalize_angle(z_bearing - self.mu[2])
            z = np.array([z_range, z_bearing_rel])
            
            # Posición esperada del landmark en coordenadas globales
            lx_obs = self.mu[0] + z_range * np.cos(z_bearing)
            ly_obs = self.mu[1] + z_range * np.sin(z_bearing)
            
            # Asociación de datos
            landmark_idx = self._data_association(z_range, z_bearing)
            
            if landmark_idx is None:
                # Nuevo landmark
                self._add_landmark(lx_obs, ly_obs)
            else:
                # Actualizar con landmark existente
                self._update_landmark(landmark_idx, z)

    def _data_association(self, z_range: float, 
                          z_bearing: float) -> Optional[int]:
        """
        Asocia una observación con un landmark existente usando
        la distancia de Mahalanobis.
        
        Returns:
            Índice del landmark asociado, o None si es nuevo
        """
        if self.num_landmarks == 0:
            return None
        
        x, y, theta = self.mu[:3]
        # Posición observada en frame global
        lx_obs = x + z_range * np.cos(z_bearing)
        ly_obs = y + z_range * np.sin(z_bearing)
        
        min_dist = float('inf')
        best_idx = None
        
        for i in range(self.num_landmarks):
            lx = self.mu[3 + 2*i]
            ly = self.mu[3 + 2*i + 1]
            
            # Distancia euclidiana simple como primera aproximación
            dist = np.hypot(lx - lx_obs, ly - ly_obs)
            
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        if min_dist < self.association_threshold:
            return best_idx
        return None

    def _add_landmark(self, lx: float, ly: float):
        """Agrega un nuevo landmark al estado."""
        # Extender estado
        self.mu = np.append(self.mu, [lx, ly])
        
        # Extender covarianza
        n = len(self.sigma)
        new_sigma = np.zeros((n + 2, n + 2))
        new_sigma[:n, :n] = self.sigma
        # Alta incertidumbre inicial para el nuevo landmark
        new_sigma[n, n] = 1.0
        new_sigma[n+1, n+1] = 1.0
        self.sigma = new_sigma
        
        self.num_landmarks += 1
        self.landmark_seen_count.append(1)

    def _update_landmark(self, idx: int, z: np.ndarray):
        """
        Actualización EKF para un landmark observado.
        
        Innovación: ν = z - ẑ
        Jacobiano: H = ∂h/∂[x, y, θ, lx, ly]
        Ganancia: K = Σ·Hᵀ·(H·Σ·Hᵀ + Q)⁻¹
        
        μ = μ + K·ν
        Σ = (I - K·H)·Σ
        """
        n = len(self.mu)
        x, y, theta = self.mu[:3]
        lx = self.mu[3 + 2*idx]
        ly = self.mu[3 + 2*idx + 1]
        
        # Medición esperada
        dx = lx - x
        dy = ly - y
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)
        
        if sqrt_q < 1e-6:
            return  # Evitar división por cero
        
        z_hat = np.array([
            sqrt_q,
            self._normalize_angle(np.arctan2(dy, dx) - theta)
        ])
        
        # Innovación
        innovation = z - z_hat
        innovation[1] = self._normalize_angle(innovation[1])
        
        # Jacobiano H (2 x n)
        H = np.zeros((2, n))
        # Derivadas respecto a la pose del robot
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, 2] = 0
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1
        
        # Derivadas respecto al landmark
        li = 3 + 2 * idx
        H[0, li] = dx / sqrt_q
        H[0, li+1] = dy / sqrt_q
        H[1, li] = -dy / q
        H[1, li+1] = dx / q
        
        # Ganancia de Kalman
        S = H @ self.sigma @ H.T + self.Q_obs
        try:
            K = self.sigma @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # Matriz singular, saltar actualización
        
        # Actualizar estado y covarianza
        self.mu += K @ innovation
        self.mu[2] = self._normalize_angle(self.mu[2])
        
        I_KH = np.eye(n) - K @ H
        self.sigma = I_KH @ self.sigma
        # Simetrizar para estabilidad numérica
        self.sigma = (self.sigma + self.sigma.T) / 2
        
        self.landmark_seen_count[idx] += 1

    def get_pose_covariance(self) -> np.ndarray:
        """Retorna la submatriz de covarianza de la pose (3x3)."""
        return self.sigma[:3, :3].copy()

    def get_landmark_covariance(self, idx: int) -> np.ndarray:
        """Retorna la covarianza de un landmark específico (2x2)."""
        li = 3 + 2 * idx
        return self.sigma[li:li+2, li:li+2].copy()

    def step(self, v: float, omega: float, dt: float,
             observations: List[Tuple[float, float]]):
        """
        Ejecuta un ciclo completo predict-update.
        
        Args:
            v, omega: controles de movimiento
            dt: paso de tiempo
            observations: lista de (rango, ángulo) de landmarks
        """
        self.predict(v, omega, dt)
        if observations:
            self.update(observations)
        self.estimated_path.append(tuple(self.mu[:3]))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normaliza ángulo al rango [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
