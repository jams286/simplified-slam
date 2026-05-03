"""
Modelo cinemático del robot diferencial.
Maneja el estado (x, y, θ) y el modelo de movimiento con ruido.
"""

import numpy as np
from typing import Tuple


class Robot:
    """
    Robot diferencial 2D con modelo de movimiento velocity-based.
    
    Estado: [x, y, θ] donde θ es la orientación en radianes.
    Control: [v, ω] velocidad lineal y angular.
    """

    def __init__(self, x: float = 10.0, y: float = 10.0, theta: float = 0.0,
                 noise_v: float = 0.02, noise_w: float = 0.01):
        self.x = x
        self.y = y
        self.theta = theta
        self.noise_v = noise_v
        self.noise_w = noise_w
        # Historial de poses reales (ground truth)
        self.true_path: list = [(x, y, theta)]

    @property
    def pose(self) -> np.ndarray:
        """Retorna el estado actual como vector [x, y, θ]."""
        return np.array([self.x, self.y, self.theta])

    def move(self, v: float, omega: float, dt: float,
             add_noise: bool = True) -> np.ndarray:
        """
        Ejecuta un paso de movimiento usando el modelo velocity motion.
        
        Modelo cinemático:
            x' = x + v·cos(θ)·dt
            y' = y + v·sin(θ)·dt
            θ' = θ + ω·dt
        
        Args:
            v: velocidad lineal (m/s)
            omega: velocidad angular (rad/s)
            dt: paso de tiempo (s)
            add_noise: si se agrega ruido de proceso
            
        Returns:
            Nuevo estado [x, y, θ]
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
        Modelo de movimiento sin ruido (para predicción en EKF).
        
        Args:
            state: [x, y, θ]
            v, omega: controles
            dt: paso de tiempo
            
        Returns:
            Estado predicho [x', y', θ']
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
        Jacobiano del modelo de movimiento respecto al estado.
        
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
        Covarianza del ruido de proceso Q.
        
        Modelo simplificado proporcional a los controles.
        """
        if alpha is None:
            alpha = np.array([0.1, 0.01, 0.01, 0.1])
        
        # Ruido en espacio de control
        M = np.diag([
            alpha[0] * v**2 + alpha[1] * omega**2,
            alpha[2] * v**2 + alpha[3] * omega**2
        ])
        
        # Jacobiano de la transformación control -> estado
        theta = 0  # Aproximación
        V = np.array([
            [np.cos(theta) * dt, 0],
            [np.sin(theta) * dt, 0],
            [0, dt]
        ])
        
        Q = V @ M @ V.T
        return Q

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normaliza ángulo al rango [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def generate_exploration_commands(num_steps: int, dt: float,
                                   env_width: float = 20.0,
                                   env_height: float = 20.0) -> list:
    """
    Genera una secuencia de comandos de movimiento para explorar el entorno.
    Usa un patrón de exploración tipo 'lawn mower' con variaciones.
    
    Returns:
        Lista de tuplas (v, omega) para cada paso.
    """
    commands = []
    phase_length = int(3.0 / dt)  # 3 segundos por fase
    turn_length = int(1.5 / dt)   # 1.5 segundos para girar
    
    v_cruise = 0.8  # velocidad crucero
    omega_turn = np.pi / 3  # velocidad de giro
    
    step = 0
    direction = 1  # 1 = derecha, -1 = izquierda
    
    while step < num_steps:
        # Avanzar recto
        for _ in range(min(phase_length, num_steps - step)):
            commands.append((v_cruise, 0.0))
            step += 1
        
        if step >= num_steps:
            break
            
        # Girar 90°
        for _ in range(min(turn_length, num_steps - step)):
            commands.append((0.2, direction * omega_turn))
            step += 1
        
        if step >= num_steps:
            break
            
        # Avanzar un poco
        short_phase = int(1.5 / dt)
        for _ in range(min(short_phase, num_steps - step)):
            commands.append((v_cruise, 0.0))
            step += 1
        
        if step >= num_steps:
            break
            
        # Girar 90° mismo sentido
        for _ in range(min(turn_length, num_steps - step)):
            commands.append((0.2, direction * omega_turn))
            step += 1
        
        direction *= -1  # Alternar dirección

    return commands[:num_steps]
