"""Tests para el módulo de entorno 2D."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment, Segment, Landmark, create_default_environment


class TestEnvironment:
    """Tests para la clase Environment."""

    def test_initialization(self):
        """Verifica la creación correcta del entorno."""
        env = Environment(10.0, 8.0)
        assert env.width == 10.0
        assert env.height == 8.0
        assert len(env.walls) == 4  # paredes perimetrales

    def test_boundary_walls(self):
        """Verifica que las paredes perimetrales son correctas."""
        env = Environment(20.0, 20.0)
        # Pared inferior: (0,0) -> (20,0)
        assert env.walls[0].x1 == 0 and env.walls[0].y1 == 0
        assert env.walls[0].x2 == 20 and env.walls[0].y2 == 0

    def test_add_rectangle(self):
        """Verifica la adición de obstáculos rectangulares."""
        env = Environment(20.0, 20.0)
        env.add_rectangle(10.0, 10.0, 2.0, 2.0)
        assert len(env.walls) == 8  # 4 perimetrales + 4 del rectángulo
        assert len(env.landmarks) == 4  # 4 esquinas

    def test_add_segment(self):
        """Verifica la adición de segmentos individuales."""
        env = Environment(20.0, 20.0)
        env.add_segment(5.0, 0.0, 5.0, 10.0)
        assert len(env.walls) == 5
        assert len(env.landmarks) == 2

    def test_ray_cast_horizontal(self):
        """Verifica ray-casting contra pared derecha."""
        env = Environment(10.0, 10.0)
        # Rayo desde el centro hacia la derecha
        dist = env.ray_cast(5.0, 5.0, 0.0, 20.0)
        assert abs(dist - 5.0) < 0.01

    def test_ray_cast_vertical(self):
        """Verifica ray-casting contra pared superior."""
        env = Environment(10.0, 10.0)
        # Rayo desde el centro hacia arriba
        dist = env.ray_cast(5.0, 5.0, np.pi/2, 20.0)
        assert abs(dist - 5.0) < 0.01

    def test_ray_cast_diagonal(self):
        """Verifica ray-casting en diagonal."""
        env = Environment(10.0, 10.0)
        dist = env.ray_cast(5.0, 5.0, np.pi/4, 20.0)
        expected = 5.0 / np.cos(np.pi/4)  # ~7.07
        assert abs(dist - expected) < 0.1

    def test_ray_cast_max_range(self):
        """Verifica que retorna max_range cuando no hay intersección cercana."""
        env = Environment(100.0, 100.0)
        dist = env.ray_cast(50.0, 50.0, 0.0, 5.0)
        assert dist == 5.0  # no alcanza la pared

    def test_is_free(self):
        """Verifica detección de colisiones."""
        env = Environment(20.0, 20.0)
        # Centro debe estar libre
        assert env.is_free(10.0, 10.0, 0.3) is True
        # Cerca del borde no debe ser libre
        assert env.is_free(0.1, 10.0, 0.3) is False

    def test_is_free_with_obstacle(self):
        """Verifica colisión con obstáculos internos."""
        env = Environment(20.0, 20.0)
        env.add_rectangle(10.0, 10.0, 2.0, 2.0)
        # Justo sobre el borde del obstáculo (dist=0 < radius)
        assert env.is_free(9.0, 10.0, 0.3) is False
        # Lejos del obstáculo
        assert env.is_free(2.0, 2.0, 0.3) is True

    def test_ground_truth_map(self):
        """Verifica generación del mapa ground truth."""
        env = Environment(10.0, 10.0)
        gt_map = env.get_ground_truth_map(resolution=0.5)
        assert gt_map.shape == (20, 20)
        # Las paredes deben estar marcadas
        assert np.sum(gt_map) > 0

    def test_default_environment(self):
        """Verifica la creación del entorno predefinido."""
        env = create_default_environment()
        assert env.width == 20.0
        assert env.height == 20.0
        assert len(env.walls) > 4  # Más que solo las perimetrales
        assert len(env.landmarks) > 0

    def test_ray_segment_intersection_parallel(self):
        """Verifica que rayos paralelos no intersectan."""
        result = Environment._ray_segment_intersection(
            0, 0, 1, 0,  # rayo horizontal
            0, 1, 5, 1   # segmento horizontal paralelo
        )
        assert result is None

    def test_ray_segment_intersection_behind(self):
        """Verifica que no detecta intersecciones detrás del rayo."""
        result = Environment._ray_segment_intersection(
            5, 5, 1, 0,   # rayo hacia derecha desde (5,5)
            0, 0, 0, 10   # segmento vertical en x=0 (detrás)
        )
        assert result is None


def run_tests():
    """Ejecuta todos los tests manualmente."""
    test = TestEnvironment()
    methods = [m for m in dir(test) if m.startswith('test_')]
    passed = 0
    failed = 0
    for method_name in methods:
        try:
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
    print("🧪 Tests: Environment")
    p, f = run_tests()
    print(f"\n   Resultado: {p} passed, {f} failed")
