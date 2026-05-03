"""
Simulador de sensor LiDAR 2D con ruido gaussiano realista.
Genera escaneos simulados mediante ray-casting en el entorno.
"""

import numpy as np
from typing import Tuple, List, Optional
from .environment import Environment


class LiDAR:
    """
    Sensor LiDAR 2D simulado con modelo de ruido realista.
    
    Características del modelo:
    - Ruido gaussiano en mediciones de distancia
    - Probabilidad de lecturas perdidas (max_range)
    - Resolución angular configurable
    - Campo de visión ajustable
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
        
        # Ángulos de los rayos relativos a la orientación del robot
        self.angles = np.linspace(-fov / 2, fov / 2, num_beams, endpoint=False)

    def scan(self, environment: Environment, robot_x: float, 
             robot_y: float, robot_theta: float) -> 'LiDARScan':
        """
        Realiza un escaneo completo del entorno desde la pose del robot.
        
        Args:
            environment: entorno con obstáculos
            robot_x, robot_y: posición del robot
            robot_theta: orientación del robot
            
        Returns:
            Objeto LiDARScan con las mediciones
        """
        ranges = np.zeros(self.num_beams)
        valid = np.ones(self.num_beams, dtype=bool)

        for i, local_angle in enumerate(self.angles):
            global_angle = robot_theta + local_angle
            
            # Ray-casting sin ruido
            true_range = environment.ray_cast(
                robot_x, robot_y, global_angle, self.max_range
            )
            
            # Modelo de ruido gaussiano
            noisy_range = true_range + np.random.normal(0, self.noise_std)
            
            # Simular lecturas perdidas
            if np.random.random() < self.miss_probability:
                noisy_range = self.max_range
                valid[i] = False
            
            # Saturar al rango válido
            noisy_range = np.clip(noisy_range, self.min_range, self.max_range)
            
            # Marcar como inválido si está en el límite
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
        Extrae landmarks (puntos notables) del escaneo usando
        detección de discontinuidades y clustering.
        
        Detecta esquinas/puntos donde hay saltos bruscos de distancia.
        
        Returns:
            Lista de (rango, ángulo) de landmarks detectados
        """
        landmarks = []
        ranges = scan.ranges
        valid = scan.valid
        
        # Detectar discontinuidades (posibles esquinas)
        for i in range(1, len(ranges) - 1):
            if not valid[i]:
                continue
            
            # Diferencia con vecinos
            if valid[i-1] and valid[i+1]:
                diff_left = abs(ranges[i] - ranges[i-1])
                diff_right = abs(ranges[i] - ranges[i+1])
                
                # Punto de discontinuidad
                if diff_left > max_cluster_gap or diff_right > max_cluster_gap:
                    landmarks.append((ranges[i], scan.angles[i]))
        
        return landmarks

    def scan_to_points(self, scan: 'LiDARScan') -> np.ndarray:
        """
        Convierte un escaneo a puntos cartesianos en el frame global.
        
        Returns:
            Array (N, 2) con coordenadas [x, y] de puntos válidos
        """
        valid_ranges = scan.ranges[scan.valid]
        valid_angles = scan.angles[scan.valid]
        
        rx, ry, _ = scan.robot_pose
        
        x = rx + valid_ranges * np.cos(valid_angles)
        y = ry + valid_ranges * np.sin(valid_angles)
        
        return np.column_stack([x, y])


class LiDARScan:
    """Resultado de un escaneo LiDAR."""
    
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
        """Número de mediciones válidas."""
        return int(np.sum(self.valid))

    def get_endpoints(self) -> np.ndarray:
        """
        Calcula los puntos finales de los rayos en coordenadas globales.
        
        Returns:
            Array (N, 2) con coordenadas [x, y] de todos los endpoints
        """
        rx, ry, _ = self.robot_pose
        x = rx + self.ranges * np.cos(self.angles)
        y = ry + self.ranges * np.sin(self.angles)
        return np.column_stack([x, y])

    def get_valid_endpoints(self) -> np.ndarray:
        """Retorna solo los endpoints de mediciones válidas."""
        endpoints = self.get_endpoints()
        return endpoints[self.valid]
