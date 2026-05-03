"""
Simulador de entorno 2D con paredes y obstáculos.
Define el mundo donde el robot navega y los elementos que el LiDAR detectará.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Segment:
    """Segmento de línea que representa una pared u obstáculo."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Landmark:
    """Punto de referencia en el entorno (esquinas, puntos notables)."""
    x: float
    y: float
    landmark_id: int


class Environment:
    """
    Entorno 2D rectangular con paredes perimetrales y obstáculos internos.
    
    El entorno se compone de segmentos de línea que representan superficies
    reflectantes para el sensor LiDAR.
    """

    def __init__(self, width: float = 20.0, height: float = 20.0):
        self.width = width
        self.height = height
        self.walls: List[Segment] = []
        self.landmarks: List[Landmark] = []
        self._build_boundary()

    def _build_boundary(self):
        """Construye las paredes perimetrales del entorno."""
        w, h = self.width, self.height
        self.walls.extend([
            Segment(0, 0, w, 0),     # pared inferior
            Segment(w, 0, w, h),     # pared derecha
            Segment(w, h, 0, h),     # pared superior
            Segment(0, h, 0, 0),     # pared izquierda
        ])

    def add_rectangle(self, cx: float, cy: float, rw: float, rh: float):
        """
        Agrega un obstáculo rectangular.
        
        Args:
            cx, cy: centro del rectángulo
            rw, rh: ancho y alto del rectángulo
        """
        x1, y1 = cx - rw / 2, cy - rh / 2
        x2, y2 = cx + rw / 2, cy + rh / 2
        self.walls.extend([
            Segment(x1, y1, x2, y1),
            Segment(x2, y1, x2, y2),
            Segment(x2, y2, x1, y2),
            Segment(x1, y2, x1, y1),
        ])
        # Agregar esquinas como landmarks
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for x, y in corners:
            lid = len(self.landmarks)
            self.landmarks.append(Landmark(x, y, lid))

    def add_segment(self, x1: float, y1: float, x2: float, y2: float):
        """Agrega un segmento de pared individual."""
        self.walls.append(Segment(x1, y1, x2, y2))
        # Endpoints como landmarks
        for x, y in [(x1, y1), (x2, y2)]:
            lid = len(self.landmarks)
            self.landmarks.append(Landmark(x, y, lid))

    def ray_cast(self, ox: float, oy: float, angle: float, 
                 max_range: float) -> float:
        """
        Lanza un rayo desde (ox, oy) en dirección 'angle' y retorna
        la distancia al primer obstáculo, o max_range si no hay intersección.
        
        Usa intersección rayo-segmento parametrizada.
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
        Calcula la intersección entre un rayo y un segmento usando
        el método parametrizado (Cramer).
        
        Retorna la distancia t si hay intersección válida, None en otro caso.
        """
        sx = x2 - x1
        sy = y2 - y1

        denom = dx * sy - dy * sx
        if abs(denom) < 1e-10:
            return None  # Rayo paralelo al segmento

        t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom

        if t >= 0 and 0 <= u <= 1:
            return t
        return None

    def is_free(self, x: float, y: float, radius: float = 0.0) -> bool:
        """Verifica si una posición es válida (sin colisión)."""
        if x - radius < 0 or x + radius > self.width:
            return False
        if y - radius < 0 or y + radius > self.height:
            return False
        # Verificar distancia a cada segmento
        for wall in self.walls[4:]:  # Saltar paredes perimetrales
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
        """Distancia mínima de un punto a un segmento."""
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
        Genera un mapa de verdad (ground truth) como grilla binaria.
        
        Returns:
            Matriz donde 1 = ocupado, 0 = libre
        """
        rows = int(self.height / resolution)
        cols = int(self.width / resolution)
        grid = np.zeros((rows, cols), dtype=np.float32)

        for wall in self.walls:
            # Rasterizar cada segmento en la grilla
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
    """Crea un entorno predefinido con obstáculos variados para demostración."""
    env = Environment(20.0, 20.0)

    # Habitaciones y pasillos
    env.add_rectangle(5.0, 5.0, 3.0, 3.0)    # obstáculo inferior izquierdo
    env.add_rectangle(15.0, 5.0, 2.0, 4.0)   # obstáculo inferior derecho
    env.add_rectangle(5.0, 15.0, 2.5, 2.5)   # obstáculo superior izquierdo
    env.add_rectangle(15.0, 15.0, 3.0, 2.0)  # obstáculo superior derecho
    env.add_rectangle(10.0, 10.0, 2.0, 2.0)  # obstáculo central

    # Paredes internas (pasillos)
    env.add_segment(8.0, 0.0, 8.0, 4.0)      # pared vertical inferior
    env.add_segment(12.0, 7.0, 12.0, 13.0)   # pared vertical central
    env.add_segment(0.0, 12.0, 4.0, 12.0)    # pared horizontal izquierda

    return env
