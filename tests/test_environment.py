"""Tests for the 2D environment module."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment, Segment, Landmark, create_default_environment


class TestEnvironment:
    """Tests for the Environment class."""

    def test_initialization(self):
        """Verifies correct environment creation."""
        env = Environment(10.0, 8.0)
        assert env.width == 10.0
        assert env.height == 8.0
        assert len(env.walls) == 4  # perimeter walls

    def test_boundary_walls(self):
        """Verifies that perimeter walls are correct."""
        env = Environment(20.0, 20.0)
        # Bottom wall: (0,0) -> (20,0)
        assert env.walls[0].x1 == 0 and env.walls[0].y1 == 0
        assert env.walls[0].x2 == 20 and env.walls[0].y2 == 0

    def test_add_rectangle(self):
        """Verifies addition of rectangular obstacles."""
        env = Environment(20.0, 20.0)
        env.add_rectangle(10.0, 10.0, 2.0, 2.0)
        assert len(env.walls) == 8  # 4 perimeter + 4 rectangle
        assert len(env.landmarks) == 4  # 4 corners

    def test_add_segment(self):
        """Verifies addition of individual segments."""
        env = Environment(20.0, 20.0)
        env.add_segment(5.0, 0.0, 5.0, 10.0)
        assert len(env.walls) == 5
        assert len(env.landmarks) == 2

    def test_ray_cast_horizontal(self):
        """Verifies ray-casting against right wall."""
        env = Environment(10.0, 10.0)
        # Ray from center to the right
        dist = env.ray_cast(5.0, 5.0, 0.0, 20.0)
        assert abs(dist - 5.0) < 0.01

    def test_ray_cast_vertical(self):
        """Verifies ray-casting against top wall."""
        env = Environment(10.0, 10.0)
        # Ray from center upwards
        dist = env.ray_cast(5.0, 5.0, np.pi/2, 20.0)
        assert abs(dist - 5.0) < 0.01

    def test_ray_cast_diagonal(self):
        """Verifies diagonal ray-casting."""
        env = Environment(10.0, 10.0)
        dist = env.ray_cast(5.0, 5.0, np.pi/4, 20.0)
        expected = 5.0 / np.cos(np.pi/4)  # ~7.07
        assert abs(dist - expected) < 0.1

    def test_ray_cast_max_range(self):
        """Verifies that max_range is returned when no nearby intersection."""
        env = Environment(100.0, 100.0)
        dist = env.ray_cast(50.0, 50.0, 0.0, 5.0)
        assert dist == 5.0  # doesn't reach the wall

    def test_is_free(self):
        """Verifies collision detection."""
        env = Environment(20.0, 20.0)
        # Center must be free
        assert env.is_free(10.0, 10.0, 0.3) is True
        # Near the edge should not be free
        assert env.is_free(0.1, 10.0, 0.3) is False

    def test_is_free_with_obstacle(self):
        """Verifies collision with internal obstacles."""
        env = Environment(20.0, 20.0)
        env.add_rectangle(10.0, 10.0, 2.0, 2.0)
        # Right on obstacle edge (dist=0 < radius)
        assert env.is_free(9.0, 10.0, 0.3) is False
        # Far from obstacle
        assert env.is_free(2.0, 2.0, 0.3) is True

    def test_ground_truth_map(self):
        """Verifies ground truth map generation."""
        env = Environment(10.0, 10.0)
        gt_map = env.get_ground_truth_map(resolution=0.5)
        assert gt_map.shape == (20, 20)
        # Walls must be marked
        assert np.sum(gt_map) > 0

    def test_default_environment(self):
        """Verifies predefined environment creation."""
        env = create_default_environment()
        assert env.width == 20.0
        assert env.height == 20.0
        assert len(env.walls) > 4  # More than just perimeter walls
        assert len(env.landmarks) > 0

    def test_ray_segment_intersection_parallel(self):
        """Verifies that parallel rays don't intersect."""
        result = Environment._ray_segment_intersection(
            0, 0, 1, 0,  # horizontal ray
            0, 1, 5, 1   # parallel horizontal segment
        )
        assert result is None

    def test_ray_segment_intersection_behind(self):
        """Verifies that intersections behind the ray are not detected."""
        result = Environment._ray_segment_intersection(
            5, 5, 1, 0,   # ray to the right from (5,5)
            0, 0, 0, 10   # vertical segment at x=0 (behind)
        )
        assert result is None


def run_tests():
    """Runs all tests manually."""
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
    print(f"\n   Result: {p} passed, {f} failed")
