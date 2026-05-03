"""Tests para el módulo de mapa de ocupación."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.occupancy_grid import OccupancyGrid
from src.lidar import LiDARScan


class TestOccupancyGrid:
    """Tests para la clase OccupancyGrid."""

    def setup(self):
        """Inicializa grilla para tests."""
        self.grid = OccupancyGrid(
            width=10.0, height=10.0, resolution=0.5
        )

    def test_initialization(self):
        """Verifica dimensiones correctas."""
        self.setup()
        assert self.grid.rows == 20
        assert self.grid.cols == 20
        assert self.grid.grid.shape == (20, 20)
        # Prior debe ser 0 (log-odds)
        assert np.allclose(self.grid.grid, 0.0)

    def test_probability_map_initial(self):
        """La probabilidad inicial debe ser 0.5 (prior uniforme)."""
        self.setup()
        prob = self.grid.get_probability_map()
        assert np.allclose(prob, 0.5)

    def test_binary_map_initial(self):
        """El mapa binario inicial debe ser todo libre con threshold > 0.5."""
        self.setup()
        binary = self.grid.get_binary_map(threshold=0.6)
        assert np.all(binary == 0)

    def test_bresenham_horizontal(self):
        """Verifica Bresenham en línea horizontal."""
        cells = OccupancyGrid._bresenham(0, 0, 5, 0)
        assert len(cells) == 6
        assert cells[0] == (0, 0)
        assert cells[-1] == (5, 0)

    def test_bresenham_vertical(self):
        """Verifica Bresenham en línea vertical."""
        cells = OccupancyGrid._bresenham(0, 0, 0, 5)
        assert len(cells) == 6
        assert cells[0] == (0, 0)
        assert cells[-1] == (0, 5)

    def test_bresenham_diagonal(self):
        """Verifica Bresenham en diagonal."""
        cells = OccupancyGrid._bresenham(0, 0, 5, 5)
        assert len(cells) >= 6
        assert cells[0] == (0, 0)
        assert cells[-1] == (5, 5)

    def test_bresenham_single_point(self):
        """Un punto a sí mismo produce una celda."""
        cells = OccupancyGrid._bresenham(3, 3, 3, 3)
        assert len(cells) == 1
        assert cells[0] == (3, 3)

    def test_update_marks_free(self):
        """La actualización marca celdas libres."""
        self.setup()
        # Simular un escaneo simple
        ranges = np.array([4.0])
        angles = np.array([0.0])
        valid = np.array([True])
        scan = LiDARScan(ranges, angles, valid, (5.0, 5.0, 0.0), 8.0)
        
        self.grid.update(scan)
        
        # Las celdas entre robot y hit deberían ser más libres
        prob = self.grid.get_probability_map()
        robot_col = int(5.0 / 0.5)
        robot_row = int(5.0 / 0.5)
        # Celda justo adelante del robot
        assert prob[robot_row, robot_col + 2] < 0.5

    def test_update_marks_occupied(self):
        """La actualización marca celdas ocupadas."""
        self.setup()
        ranges = np.array([3.0])
        angles = np.array([0.0])
        valid = np.array([True])
        scan = LiDARScan(ranges, angles, valid, (5.0, 5.0, 0.0), 8.0)
        
        self.grid.update(scan)
        
        # La celda final (hit) debería ser más ocupada
        prob = self.grid.get_probability_map()
        hit_col = int(8.0 / 0.5)
        hit_row = int(5.0 / 0.5)
        if 0 <= hit_row < self.grid.rows and 0 <= hit_col < self.grid.cols:
            assert prob[hit_row, hit_col] > 0.5

    def test_log_odds_saturation(self):
        """Log-odds no deben exceder los límites."""
        self.setup()
        # Múltiples actualizaciones para saturar
        ranges = np.array([3.0])
        angles = np.array([0.0])
        valid = np.array([True])
        scan = LiDARScan(ranges, angles, valid, (5.0, 5.0, 0.0), 8.0)
        
        for _ in range(100):
            self.grid.update(scan)
        
        assert np.all(self.grid.grid >= self.grid.log_odd_min)
        assert np.all(self.grid.grid <= self.grid.log_odd_max)

    def test_explored_mask(self):
        """Verifica que la máscara de exploración se actualiza."""
        self.setup()
        assert np.sum(self.grid.get_explored_mask()) == 0
        
        ranges = np.array([3.0, 4.0, 5.0])
        angles = np.array([0.0, np.pi/4, np.pi/2])
        valid = np.array([True, True, True])
        scan = LiDARScan(ranges, angles, valid, (5.0, 5.0, 0.0), 8.0)
        
        self.grid.update(scan)
        assert np.sum(self.grid.get_explored_mask()) > 0

    def test_world_to_grid(self):
        """Verifica conversión mundo -> grilla."""
        self.setup()
        col, row = self.grid.world_to_grid(5.0, 5.0)
        assert col == 10
        assert row == 10

    def test_grid_to_world(self):
        """Verifica conversión grilla -> mundo."""
        self.setup()
        x, y = self.grid.to_world(10, 10) if hasattr(self.grid, 'to_world') else self.grid.grid_to_world(10, 10)
        assert abs(x - 5.25) < 0.01
        assert abs(y - 5.25) < 0.01

    def test_update_from_pose(self):
        """Verifica actualización con pose estimada."""
        self.setup()
        ranges = np.array([3.0, 4.0])
        angles = np.array([0.0, np.pi/2])
        valid = np.array([True, True])
        scan = LiDARScan(ranges, angles, valid, (5.0, 5.0, 0.0), 8.0)
        
        estimated_pose = np.array([5.1, 5.1, 0.0])
        self.grid.update_from_pose(scan, estimated_pose)
        
        # Debe haber actualizado celdas
        assert np.sum(self.grid.get_explored_mask()) > 0


def run_tests():
    """Ejecuta todos los tests."""
    test = TestOccupancyGrid()
    methods = [m for m in dir(test) if m.startswith('test_')]
    passed = 0
    failed = 0
    for method_name in methods:
        try:
            test.setup()
            getattr(test, method_name)()
            print(f"  ✅ {method_name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {method_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {method_name}: {type(e).__name__}: {e}")
            failed += 1
    return passed, failed


if __name__ == '__main__':
    print("🧪 Tests: Occupancy Grid")
    p, f = run_tests()
    print(f"\n   Resultado: {p} passed, {f} failed")
