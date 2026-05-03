"""
Runner de tests para todos los módulos.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_environment import run_tests as run_env_tests
from tests.test_lidar import run_tests as run_lidar_tests
from tests.test_ekf_slam import run_tests as run_ekf_tests
from tests.test_occupancy_grid import run_tests as run_grid_tests
from tests.test_error_analysis import run_tests as run_error_tests


def main():
    print("=" * 60)
    print("   🧪 SLAM SIMPLIFICADO - Suite de Tests")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    results = []
    
    suites = [
        ("Environment", run_env_tests),
        ("LiDAR", run_lidar_tests),
        ("EKF-SLAM", run_ekf_tests),
        ("Occupancy Grid", run_grid_tests),
        ("Error Analysis", run_error_tests),
    ]
    
    for name, runner in suites:
        print(f"\n{'─' * 40}")
        print(f"  📋 {name}")
        print(f"{'─' * 40}")
        try:
            p, f = runner()
            total_passed += p
            total_failed += f
            results.append((name, p, f))
        except Exception as e:
            print(f"  ❌ Error ejecutando suite: {e}")
            total_failed += 1
            results.append((name, 0, 1))
    
    # Resumen
    print(f"\n{'=' * 60}")
    print(f"   📊 RESUMEN DE RESULTADOS")
    print(f"{'=' * 60}")
    print(f"\n   {'Suite':<20} {'Passed':<10} {'Failed':<10}")
    print(f"   {'─'*40}")
    for name, p, f in results:
        status = "✅" if f == 0 else "❌"
        print(f"   {status} {name:<18} {p:<10} {f:<10}")
    
    print(f"\n   {'─'*40}")
    print(f"   Total: {total_passed} passed, {total_failed} failed")
    print(f"   Tasa de éxito: {total_passed/(total_passed+total_failed)*100:.1f}%")
    print(f"{'=' * 60}")
    
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
