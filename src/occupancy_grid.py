"""
Mapa de grilla de ocupación (Occupancy Grid Map).
Implementa el modelo de sensor inverso con log-odds para
actualización eficiente del mapa.
"""

import numpy as np
from typing import Tuple
from .lidar import LiDARScan


class OccupancyGrid:
    """
    Mapa de grilla de ocupación usando representación log-odds.
    
    Modelo probabilístico:
        L(m|z) = L(m|z₁:ₜ₋₁) + L(m|zₜ) - L₀
        
    Donde L = log(p/(1-p)) es la transformación log-odds.
    
    Ventajas del log-odds:
    - Actualización aditiva (eficiente)
    - Evita problemas numéricos con probabilidades cercanas a 0 o 1
    - Saturación natural con límites
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
        
        # Dimensiones de la grilla
        self.cols = int(width / resolution)
        self.rows = int(height / resolution)
        
        # Grilla en log-odds (inicializada con prior)
        self.grid = np.full((self.rows, self.cols), log_odd_prior, 
                           dtype=np.float32)
        
        # Contador de actualizaciones por celda
        self.update_count = np.zeros((self.rows, self.cols), dtype=np.int32)

    def update(self, scan: LiDARScan):
        """
        Actualiza el mapa con un escaneo LiDAR usando el modelo
        de sensor inverso (inverse sensor model).
        
        Para cada rayo:
        - Celdas atravesadas → incrementar log-odds libre
        - Celda final (hit) → incrementar log-odds ocupado
        
        Usa Bresenham para trazar los rayos eficientemente.
        """
        rx, ry, _ = scan.robot_pose
        robot_col = int(rx / self.resolution)
        robot_row = int(ry / self.resolution)
        
        endpoints = scan.get_endpoints()
        
        for i in range(len(scan.ranges)):
            ex, ey = endpoints[i]
            end_col = int(ex / self.resolution)
            end_row = int(ey / self.resolution)
            
            # Trazar rayo con Bresenham
            cells = self._bresenham(robot_col, robot_row, end_col, end_row)
            
            # Celdas libres (todas excepto la última)
            for col, row in cells[:-1]:
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    self.grid[row, col] += self.log_odd_free
                    self.grid[row, col] = max(self.grid[row, col], 
                                             self.log_odd_min)
                    self.update_count[row, col] += 1
            
            # Celda ocupada (última celda, solo si el rayo impactó)
            if scan.valid[i] and cells:
                col, row = cells[-1]
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    self.grid[row, col] += self.log_odd_occ
                    self.grid[row, col] = min(self.grid[row, col], 
                                             self.log_odd_max)
                    self.update_count[row, col] += 1

    def update_from_pose(self, scan: LiDARScan, estimated_pose: np.ndarray):
        """
        Actualiza el mapa usando la pose estimada por SLAM en lugar
        de la pose real del robot.
        
        Args:
            scan: escaneo LiDAR original
            estimated_pose: [x, y, θ] estimado por EKF-SLAM
        """
        rx, ry, rtheta = estimated_pose
        robot_col = int(rx / self.resolution)
        robot_row = int(ry / self.resolution)
        
        for i in range(len(scan.ranges)):
            if not scan.valid[i]:
                continue
                
            # Recalcular endpoint con pose estimada
            angle = scan.angles[i] - scan.robot_pose[2] + rtheta
            ex = rx + scan.ranges[i] * np.cos(angle)
            ey = ry + scan.ranges[i] * np.sin(angle)
            
            end_col = int(ex / self.resolution)
            end_row = int(ey / self.resolution)
            
            # Trazar rayo
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
        Convierte log-odds a probabilidades de ocupación [0, 1].
        
        p = 1 - 1/(1 + exp(L))
        """
        return 1.0 - 1.0 / (1.0 + np.exp(self.grid))

    def get_binary_map(self, threshold: float = 0.6) -> np.ndarray:
        """
        Genera mapa binario: 1 = ocupado, 0 = libre.
        
        Args:
            threshold: umbral de probabilidad para considerar ocupado
        """
        prob_map = self.get_probability_map()
        return (prob_map > threshold).astype(np.float32)

    def get_explored_mask(self) -> np.ndarray:
        """Retorna máscara de celdas que han sido observadas al menos una vez."""
        return self.update_count > 0

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list:
        """
        Algoritmo de Bresenham para trazar una línea en la grilla.
        
        Returns:
            Lista de tuplas (col, row) de celdas atravesadas.
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
        """Convierte coordenadas del mundo a índices de grilla."""
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convierte índices de grilla a coordenadas del mundo."""
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return x, y
