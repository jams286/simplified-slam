"""Tests for the EKF-SLAM module."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ekf_slam import EKFSLAM


class TestEKFSLAM:
    """Tests for the EKFSLAM class."""

    def setup(self):
        """Initializes EKF-SLAM for tests."""
        self.initial_pose = np.array([5.0, 5.0, 0.0])
        self.ekf = EKFSLAM(
            initial_pose=self.initial_pose,
            range_noise=0.1,
            bearing_noise=0.05,
            association_threshold=1.5
        )

    def test_initialization(self):
        """Verifies correct initialization."""
        self.setup()
        assert np.allclose(self.ekf.robot_pose, self.initial_pose)
        assert self.ekf.num_landmarks == 0
        assert self.ekf.sigma.shape == (3, 3)

    def test_predict_straight(self):
        """Verifies straight motion prediction."""
        self.setup()
        self.ekf.predict(v=1.0, omega=0.0, dt=1.0)
        # Should move 1m in x (theta=0)
        pose = self.ekf.robot_pose
        assert abs(pose[0] - 6.0) < 0.01
        assert abs(pose[1] - 5.0) < 0.01
        assert abs(pose[2]) < 0.01

    def test_predict_turn(self):
        """Verifies turn prediction."""
        self.setup()
        self.ekf.predict(v=0.0, omega=np.pi/2, dt=1.0)
        pose = self.ekf.robot_pose
        assert abs(pose[0] - 5.0) < 0.01  # Does not move in x
        assert abs(pose[1] - 5.0) < 0.01  # Does not move in y
        assert abs(pose[2] - np.pi/2) < 0.01  # Turns 90°

    def test_predict_increases_uncertainty(self):
        """Prediction must increase covariance."""
        self.setup()
        cov_before = np.trace(self.ekf.sigma)
        self.ekf.predict(v=1.0, omega=0.5, dt=0.1)
        cov_after = np.trace(self.ekf.sigma)
        assert cov_after > cov_before

    def test_add_landmark(self):
        """Verifies that landmarks are added correctly."""
        self.setup()
        # Observe a landmark at 3m, 0rad (in front of robot at theta=0)
        observations = [(3.0, 0.0)]  # range=3, angle=0 (global)
        self.ekf.update(observations)
        assert self.ekf.num_landmarks == 1
        # Landmark should be at ~(8, 5)
        lm = self.ekf.landmarks[0]
        assert abs(lm[0] - 8.0) < 0.5
        assert abs(lm[1] - 5.0) < 0.5

    def test_state_dimension_grows(self):
        """State grows with each new landmark."""
        self.setup()
        assert len(self.ekf.mu) == 3
        self.ekf.update([(3.0, 0.0)])
        assert len(self.ekf.mu) == 5
        self.ekf.update([(4.0, np.pi/2)])
        assert len(self.ekf.mu) == 7

    def test_covariance_symmetry(self):
        """Covariance must be symmetric."""
        self.setup()
        self.ekf.predict(v=1.0, omega=0.1, dt=0.1)
        self.ekf.update([(3.0, 0.5)])
        self.ekf.predict(v=0.5, omega=-0.2, dt=0.1)
        self.ekf.update([(2.0, -0.3)])
        
        assert np.allclose(self.ekf.sigma, self.ekf.sigma.T, atol=1e-10)

    def test_data_association(self):
        """Verifies that nearby landmarks are associated correctly."""
        self.setup()
        # First landmark
        self.ekf.update([(3.0, 0.0)])
        n_before = self.ekf.num_landmarks
        # Observe the same landmark again (similar position)
        self.ekf.update([(3.0, 0.0)])
        # Should not add a new landmark
        assert self.ekf.num_landmarks == n_before

    def test_new_landmark_far(self):
        """A far landmark must create a new one."""
        self.setup()
        self.ekf.update([(3.0, 0.0)])
        n_before = self.ekf.num_landmarks
        # Observe in completely different direction
        self.ekf.update([(5.0, np.pi)])
        assert self.ekf.num_landmarks == n_before + 1

    def test_step_method(self):
        """Verifies the step method (predict + update)."""
        self.setup()
        self.ekf.step(v=1.0, omega=0.0, dt=0.1, observations=[(3.0, 0.2)])
        assert len(self.ekf.estimated_path) == 2
        assert self.ekf.num_landmarks >= 1

    def test_multiple_steps_convergence(self):
        """Multiple observations of the same landmark reduce uncertainty."""
        self.setup()
        # Observe a fixed landmark multiple times
        for _ in range(10):
            self.ekf.step(v=0.0, omega=0.0, dt=0.1, observations=[(3.0, 0.0)])
        
        if self.ekf.num_landmarks > 0:
            lm_cov = self.ekf.get_landmark_covariance(0)
            # Covariance should be small after many observations
            assert np.trace(lm_cov) < 1.0

    def test_normalize_angle(self):
        """Verifies angle normalization."""
        assert abs(EKFSLAM._normalize_angle(3*np.pi) - np.pi) < 0.01
        assert abs(EKFSLAM._normalize_angle(-3*np.pi) + np.pi) < 0.01
        assert abs(EKFSLAM._normalize_angle(0.5)) < np.pi

    def test_pose_covariance(self):
        """Verifies that get_pose_covariance returns 3x3."""
        self.setup()
        cov = self.ekf.get_pose_covariance()
        assert cov.shape == (3, 3)

    def test_estimated_path_tracking(self):
        """Verifies that pose history is updated."""
        self.setup()
        for i in range(5):
            self.ekf.step(v=1.0, omega=0.0, dt=0.1, observations=[])
        assert len(self.ekf.estimated_path) == 6  # initial + 5 steps


def run_tests():
    """Runs all tests."""
    test = TestEKFSLAM()
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
    print("🧪 Tests: EKF-SLAM")
    p, f = run_tests()
    print(f"\n   Result: {p} passed, {f} failed")
