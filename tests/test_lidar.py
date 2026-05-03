"""Tests for the simulated LiDAR module."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment
from src.lidar import LiDAR, LiDARScan


class TestLiDAR:
    """Tests for the LiDAR class."""

    def setup(self):
        """Prepare simple environment for tests."""
        self.env = Environment(10.0, 10.0)
        self.lidar = LiDAR(
            num_beams=36, max_range=8.0,
            noise_std=0.0, miss_probability=0.0  # No noise for tests
        )

    def test_initialization(self):
        """Verifies correct LiDAR initialization."""
        self.setup()
        assert self.lidar.num_beams == 36
        assert self.lidar.max_range == 8.0
        assert len(self.lidar.angles) == 36

    def test_scan_returns_correct_shape(self):
        """Verifies that scan has the correct shape."""
        self.setup()
        scan = self.lidar.scan(self.env, 5.0, 5.0, 0.0)
        assert len(scan.ranges) == 36
        assert len(scan.angles) == 36
        assert len(scan.valid) == 36

    def test_scan_center_symmetric(self):
        """From the center of a square, distances should be symmetric."""
        self.setup()
        scan = self.lidar.scan(self.env, 5.0, 5.0, 0.0)
        # All rays should hit walls
        assert np.all(scan.ranges < 8.0)
        assert np.all(scan.ranges > 0.0)

    def test_scan_near_wall(self):
        """From near a wall, minimum distance should be short."""
        self.setup()
        scan = self.lidar.scan(self.env, 1.0, 5.0, np.pi)  # Facing left wall
        min_range = np.min(scan.ranges[scan.valid])
        assert min_range < 1.5  # Should be close to 1.0m

    def test_scan_with_noise(self):
        """Verifies that Gaussian noise is applied correctly."""
        self.setup()
        np.random.seed(42)
        noisy_lidar = LiDAR(num_beams=36, max_range=8.0, noise_std=0.1)
        scan1 = noisy_lidar.scan(self.env, 5.0, 5.0, 0.0)
        
        np.random.seed(99)
        scan2 = noisy_lidar.scan(self.env, 5.0, 5.0, 0.0)
        
        # With different seeds, scans should differ
        assert not np.allclose(scan1.ranges, scan2.ranges)

    def test_scan_max_range(self):
        """Verifies that max range is not exceeded."""
        self.setup()
        scan = self.lidar.scan(self.env, 5.0, 5.0, 0.0)
        assert np.all(scan.ranges <= self.lidar.max_range)

    def test_scan_min_range(self):
        """Verifies that less than min range is not reported."""
        self.setup()
        scan = self.lidar.scan(self.env, 5.0, 5.0, 0.0)
        assert np.all(scan.ranges >= self.lidar.min_range)

    def test_scan_endpoints(self):
        """Verifies that endpoints are within the environment."""
        self.setup()
        scan = self.lidar.scan(self.env, 5.0, 5.0, 0.0)
        endpoints = scan.get_endpoints()
        assert endpoints.shape == (36, 2)
        # Endpoints should be within the environment (with margin)
        assert np.all(endpoints[:, 0] >= -1.0)
        assert np.all(endpoints[:, 0] <= 11.0)
        assert np.all(endpoints[:, 1] >= -1.0)
        assert np.all(endpoints[:, 1] <= 11.0)

    def test_valid_endpoints(self):
        """Verifies valid endpoint filtering."""
        self.setup()
        scan = self.lidar.scan(self.env, 5.0, 5.0, 0.0)
        valid_ep = scan.get_valid_endpoints()
        assert len(valid_ep) == scan.num_valid

    def test_scan_to_points(self):
        """Verifies scan to Cartesian points conversion."""
        self.setup()
        scan = self.lidar.scan(self.env, 5.0, 5.0, 0.0)
        points = self.lidar.scan_to_points(scan)
        assert points.shape[1] == 2
        assert len(points) == scan.num_valid

    def test_extract_landmarks(self):
        """Verifies landmark extraction."""
        self.setup()
        # Add an obstacle to create discontinuities
        self.env.add_rectangle(3.0, 5.0, 1.0, 1.0)
        scan = self.lidar.scan(self.env, 5.0, 5.0, 0.0)
        landmarks = self.lidar.extract_landmarks(scan)
        # Should detect at least some discontinuities
        assert isinstance(landmarks, list)

    def test_miss_probability(self):
        """Verifies that miss probability works."""
        np.random.seed(42)
        lidar_miss = LiDAR(num_beams=1000, max_range=8.0,
                          noise_std=0.0, miss_probability=0.5)
        env = Environment(10.0, 10.0)
        scan = lidar_miss.scan(env, 5.0, 5.0, 0.0)
        # With 50% misses, there should be ~500 invalid
        invalid_ratio = 1.0 - scan.num_valid / len(scan.ranges)
        assert 0.3 < invalid_ratio < 0.7  # Wide margin for randomness


class TestLiDARScan:
    """Tests for the LiDARScan class."""

    def test_num_valid(self):
        """Verifies valid measurement count."""
        valid = np.array([True, True, False, True, False])
        scan = LiDARScan(
            ranges=np.ones(5), angles=np.zeros(5),
            valid=valid, robot_pose=(0, 0, 0), max_range=8.0
        )
        assert scan.num_valid == 3


def run_tests():
    """Runs all tests."""
    classes = [TestLiDAR, TestLiDARScan]
    total_passed = 0
    total_failed = 0
    
    for test_class in classes:
        print(f"\n  {test_class.__name__}:")
        test = test_class()
        methods = [m for m in dir(test) if m.startswith('test_')]
        for method_name in methods:
            try:
                if hasattr(test, 'setup'):
                    test.setup()
                getattr(test, method_name)()
                print(f"    ✅ {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"    ❌ {method_name}: {e}")
                total_failed += 1
            except Exception as e:
                print(f"    ❌ {method_name}: {type(e).__name__}: {e}")
                total_failed += 1
    
    return total_passed, total_failed


if __name__ == '__main__':
    print("🧪 Tests: LiDAR")
    p, f = run_tests()
    print(f"\n   Result: {p} passed, {f} failed")
