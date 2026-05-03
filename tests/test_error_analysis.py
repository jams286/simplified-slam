"""Tests for the error analysis module."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.error_analysis import ErrorAnalysis


class TestErrorAnalysis:
    """Tests for the ErrorAnalysis class."""

    def test_ate_zero_error(self):
        """ATE should be 0 with identical trajectories."""
        path = [(i, i, 0.0) for i in range(10)]
        result = ErrorAnalysis.absolute_trajectory_error(path, path)
        assert result['rmse'] < 1e-10
        assert result['mean'] < 1e-10

    def test_ate_known_error(self):
        """ATE with known constant error."""
        true_path = [(i, 0, 0) for i in range(10)]
        est_path = [(i, 1.0, 0) for i in range(10)]  # offset de 1m en y
        result = ErrorAnalysis.absolute_trajectory_error(true_path, est_path)
        assert abs(result['rmse'] - 1.0) < 0.01
        assert abs(result['mean'] - 1.0) < 0.01

    def test_ate_increasing_error(self):
        """ATE with increasing error."""
        true_path = [(i, 0, 0) for i in range(10)]
        est_path = [(i, i*0.1, 0) for i in range(10)]
        result = ErrorAnalysis.absolute_trajectory_error(true_path, est_path)
        assert result['max'] > result['min']
        assert result['rmse'] > 0

    def test_orientation_error_zero(self):
        """Zero orientation error with identical trajectories."""
        path = [(0, 0, i*0.1) for i in range(10)]
        result = ErrorAnalysis.orientation_error(path, path)
        assert result['rmse'] < 1e-10

    def test_orientation_error_constant(self):
        """Constant orientation error."""
        true_path = [(0, 0, i*0.1) for i in range(10)]
        est_path = [(0, 0, i*0.1 + 0.1) for i in range(10)]
        result = ErrorAnalysis.orientation_error(true_path, est_path)
        assert abs(result['mean'] - 0.1) < 0.01

    def test_map_accuracy_perfect(self):
        """Perfect accuracy with identical maps."""
        gt = np.zeros((10, 10))
        gt[5, :] = 1.0  # One occupied row
        result = ErrorAnalysis.map_accuracy(gt, gt)
        assert result['accuracy'] == 1.0
        assert result['f1_score'] == 1.0

    def test_map_accuracy_empty(self):
        """Completely empty maps."""
        gt = np.zeros((10, 10))
        est = np.zeros((10, 10))
        result = ErrorAnalysis.map_accuracy(gt, est)
        assert result['accuracy'] == 1.0

    def test_map_accuracy_all_wrong(self):
        """Minimum accuracy with inverted maps."""
        gt = np.zeros((10, 10))
        gt[5, :] = 1.0
        est = 1.0 - gt
        result = ErrorAnalysis.map_accuracy(gt, est)
        assert result['accuracy'] < 0.5

    def test_map_precision_recall(self):
        """Verifies precision and recall."""
        gt = np.zeros((10, 10))
        gt[5, 5] = 1.0
        gt[5, 6] = 1.0
        
        est = np.zeros((10, 10))
        est[5, 5] = 1.0  # TP
        est[5, 7] = 1.0  # FP
        
        result = ErrorAnalysis.map_accuracy(gt, est)
        # Precision: 1/(1+1) = 0.5
        assert abs(result['precision'] - 0.5) < 0.01
        # Recall: 1/(1+1) = 0.5
        assert abs(result['recall'] - 0.5) < 0.01

    def test_map_iou(self):
        """Verifies IoU calculation."""
        gt = np.zeros((10, 10))
        gt[0:5, 0:5] = 1.0  # 25 celdas
        
        est = np.zeros((10, 10))
        est[0:5, 0:5] = 1.0  # Perfecta coincidencia
        
        result = ErrorAnalysis.map_accuracy(gt, est)
        assert abs(result['iou'] - 1.0) < 0.01

    def test_landmark_error_perfect(self):
        """Landmark error with exact positions."""
        true_lm = np.array([[1.0, 2.0], [3.0, 4.0]])
        est_lm = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ErrorAnalysis.landmark_error(true_lm, est_lm)
        assert result['mean_error'] < 0.01
        assert result['num_matched'] == 2

    def test_landmark_error_offset(self):
        """Landmark error with known offset."""
        true_lm = np.array([[1.0, 1.0], [5.0, 5.0]])
        est_lm = np.array([[1.1, 1.1], [5.1, 5.1]])
        result = ErrorAnalysis.landmark_error(true_lm, est_lm)
        expected_error = np.sqrt(0.1**2 + 0.1**2)
        assert abs(result['mean_error'] - expected_error) < 0.01

    def test_landmark_error_empty(self):
        """Error with empty landmarks."""
        true_lm = np.array([[1.0, 2.0]])
        est_lm = np.empty((0, 2))
        result = ErrorAnalysis.landmark_error(true_lm, est_lm)
        assert result['num_matched'] == 0

    def test_exploration_coverage(self):
        """Verifies coverage calculation."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[:50, :] = True  # 50% explored
        result = ErrorAnalysis.exploration_coverage(mask)
        assert abs(result['coverage_ratio'] - 0.5) < 0.01

    def test_generate_report(self):
        """Verifies that a valid report is generated."""
        traj = {'rmse': 0.1, 'mean': 0.08, 'max': 0.2, 'std': 0.03, 'errors': np.array([0.1])}
        orient = {'rmse': 0.05, 'mean': 0.04, 'max': 0.1, 'errors': np.array([0.05])}
        maps = {'accuracy': 0.95, 'precision': 0.8, 'recall': 0.7, 'f1_score': 0.75, 'iou': 0.6,
                'true_positives': 100, 'true_negatives': 900, 'false_positives': 25, 'false_negatives': 43, 'total_cells': 1068}
        lm = {'mean_error': 0.3, 'num_estimated': 10, 'num_true': 12, 'match_rate': 0.83}
        cov = {'coverage_ratio': 0.75, 'explored_cells': 7500, 'total_cells': 10000}
        
        report = ErrorAnalysis.generate_report(traj, orient, maps, lm, cov)
        assert 'RMSE' in report
        assert 'Accuracy' in report
        assert len(report) > 100


def run_tests():
    """Runs all tests."""
    test = TestErrorAnalysis()
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
    print("🧪 Tests: Error Analysis")
    p, f = run_tests()
    print(f"\n   Result: {p} passed, {f} failed")
