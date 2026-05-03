"""
Main entry point for the simplified SLAM system.
Orchestrates all modules: environment, robot, LiDAR, EKF-SLAM,
occupancy grid, visualization, and error analysis.
"""

import numpy as np
import sys
import os
import argparse

from config import (
    ENV_WIDTH, ENV_HEIGHT, DT, LIDAR_NUM_BEAMS, LIDAR_MAX_RANGE,
    LIDAR_MIN_RANGE, LIDAR_FOV, LIDAR_NOISE_STD, LIDAR_MISS_PROB,
    GRID_RESOLUTION, GRID_LOG_ODD_FREE, GRID_LOG_ODD_OCC,
    GRID_LOG_ODD_PRIOR, GRID_LOG_ODD_MAX, GRID_LOG_ODD_MIN,
    EKF_RANGE_NOISE, EKF_BEARING_NOISE, EKF_ASSOCIATION_THRESHOLD,
    SIM_STEPS, SIM_SEED, VIS_FIGSIZE, MOTION_NOISE_V, MOTION_NOISE_W
)
from src.environment import Environment, create_default_environment
from src.robot import Robot, generate_exploration_commands
from src.lidar import LiDAR
from src.ekf_slam import EKFSLAM
from src.occupancy_grid import OccupancyGrid
from src.visualization import SLAMVisualizer
from src.error_analysis import ErrorAnalysis


def run_slam(num_steps: int = SIM_STEPS, seed: int = SIM_SEED,
             visualize: bool = True, animate: bool = True,
             save_animation: str = None):
    """
    Runs the complete SLAM simulation.
    
    Args:
        num_steps: number of simulation steps
        seed: seed for reproducibility
        visualize: whether to show final visualization
        animate: whether to generate real-time animation
        save_animation: path to save animation (GIF)
    """
    np.random.seed(seed)
    print("🤖 Simplified SLAM - EKF-SLAM with 2D LiDAR")
    print("=" * 50)
    
    # === 1. Create environment ===
    print("\n📦 Creating 2D environment...")
    env = create_default_environment()
    print(f"   Dimensions: {env.width}x{env.height} m")
    print(f"   Walls: {len(env.walls)}")
    print(f"   Landmarks: {len(env.landmarks)}")
    
    # === 2. Initialize robot ===
    initial_pose = np.array([2.0, 2.0, 0.0])
    robot = Robot(
        x=initial_pose[0], y=initial_pose[1], theta=initial_pose[2],
        noise_v=MOTION_NOISE_V, noise_w=MOTION_NOISE_W
    )
    print(f"\n🤖 Robot initialized at ({robot.x:.1f}, {robot.y:.1f}, {np.degrees(robot.theta):.0f}°)")
    
    # === 3. Initialize LiDAR ===
    lidar = LiDAR(
        num_beams=LIDAR_NUM_BEAMS, max_range=LIDAR_MAX_RANGE,
        min_range=LIDAR_MIN_RANGE, fov=LIDAR_FOV,
        noise_std=LIDAR_NOISE_STD, miss_probability=LIDAR_MISS_PROB
    )
    print(f"📡 LiDAR: {LIDAR_NUM_BEAMS} beams, max range {LIDAR_MAX_RANGE}m, noise σ={LIDAR_NOISE_STD}m")
    
    # === 4. Initialize EKF-SLAM ===
    ekf = EKFSLAM(
        initial_pose=initial_pose,
        range_noise=EKF_RANGE_NOISE,
        bearing_noise=EKF_BEARING_NOISE,
        association_threshold=EKF_ASSOCIATION_THRESHOLD
    )
    print(f"🧮 EKF-SLAM: R_noise={EKF_RANGE_NOISE}, B_noise={EKF_BEARING_NOISE}")
    
    # === 5. Initialize occupancy grid ===
    occ_grid = OccupancyGrid(
        width=ENV_WIDTH, height=ENV_HEIGHT, resolution=GRID_RESOLUTION,
        log_odd_free=GRID_LOG_ODD_FREE, log_odd_occ=GRID_LOG_ODD_OCC,
        log_odd_prior=GRID_LOG_ODD_PRIOR, log_odd_max=GRID_LOG_ODD_MAX,
        log_odd_min=GRID_LOG_ODD_MIN
    )
    print(f"🗺️  Occupancy grid: {occ_grid.rows}x{occ_grid.cols} cells ({GRID_RESOLUTION}m/cell)")
    
    # === 6. Initialize visualizer ===
    vis = SLAMVisualizer(env, occ_grid, figsize=VIS_FIGSIZE)
    
    # === 7. Generate exploration commands ===
    commands = generate_exploration_commands(num_steps, DT, ENV_WIDTH, ENV_HEIGHT)
    print(f"\n🎯 Starting exploration: {num_steps} steps ({num_steps*DT:.1f}s)")
    print("-" * 50)
    
    # === 8. Main simulation loop ===
    landmarks_true = np.array([[lm.x, lm.y] for lm in env.landmarks])
    frame_skip = max(1, num_steps // 100)  # Limit frames for performance
    
    for step in range(num_steps):
        v, omega = commands[step]
        
        # Move robot (with noise)
        robot.move(v, omega, DT, add_noise=True)
        
        # Check collision — if not free, revert
        if not env.is_free(robot.x, robot.y, radius=0.3):
            robot.x = robot.true_path[-2][0] if len(robot.true_path) > 1 else robot.x
            robot.y = robot.true_path[-2][1] if len(robot.true_path) > 1 else robot.y
            robot.true_path[-1] = (robot.x, robot.y, robot.theta)
            continue
        
        # LiDAR scan
        scan = lidar.scan(env, robot.x, robot.y, robot.theta)
        
        # Extract landmarks from scan
        landmark_obs = lidar.extract_landmarks(scan)
        
        # EKF-SLAM step
        ekf.step(v, omega, DT, landmark_obs)
        
        # Update occupancy grid with estimated pose
        estimated_pose = ekf.robot_pose
        occ_grid.update_from_pose(scan, estimated_pose)
        
        # Record frame for animation (every N steps)
        if step % frame_skip == 0:
            vis.record_frame(
                true_pose=robot.pose,
                estimated_pose=estimated_pose,
                scan_endpoints=scan.get_valid_endpoints(),
                landmarks_true=landmarks_true,
                landmarks_estimated=ekf.landmarks,
                true_path=robot.true_path,
                estimated_path=ekf.estimated_path,
                pose_covariance=ekf.get_pose_covariance()
            )
        
        # Progress
        if (step + 1) % (num_steps // 10) == 0:
            ate = np.linalg.norm(robot.pose[:2] - estimated_pose[:2])
            print(f"   Step {step+1:4d}/{num_steps} | "
                  f"Landmarks: {ekf.num_landmarks:3d} | "
                  f"ATE: {ate:.4f}m | "
                  f"Coverage: {np.mean(occ_grid.get_explored_mask()):.1%}")
    
    print("-" * 50)
    print("✅ Simulation completed")
    
    # === 9. Error analysis ===
    print("\n📊 Computing error metrics...")
    
    gt_map = env.get_ground_truth_map(GRID_RESOLUTION)
    estimated_binary = occ_grid.get_binary_map(threshold=0.6)
    explored_mask = occ_grid.get_explored_mask()
    
    traj_error = ErrorAnalysis.absolute_trajectory_error(
        robot.true_path, ekf.estimated_path
    )
    orient_error = ErrorAnalysis.orientation_error(
        robot.true_path, ekf.estimated_path
    )
    map_metrics = ErrorAnalysis.map_accuracy(
        gt_map, estimated_binary, explored_mask
    )
    lm_error = ErrorAnalysis.landmark_error(
        landmarks_true, ekf.landmarks
    )
    coverage = ErrorAnalysis.exploration_coverage(explored_mask)
    
    # Print report
    report = ErrorAnalysis.generate_report(
        traj_error, orient_error, map_metrics, lm_error, coverage
    )
    print(report)
    
    # === 10. Visualization ===
    if save_animation:
        os.makedirs(os.path.dirname(save_animation), exist_ok=True)

    if visualize:
        print("\n🎨 Generating final visualization...")
        save_comparison = None
        if save_animation:
            results_dir = os.path.dirname(save_animation) or 'results'
            os.makedirs(results_dir, exist_ok=True)
            save_comparison = os.path.join(results_dir, 'comparison.png')
        vis.plot_final_comparison(
            robot.true_path, ekf.estimated_path, gt_map,
            save_path=save_comparison
        )
    
    if animate:
        print("🎬 Generating animation...")
        vis.animate(interval=100, save_path=save_animation)
    
    return {
        'trajectory_error': traj_error,
        'orientation_error': orient_error,
        'map_metrics': map_metrics,
        'landmark_error': lm_error,
        'coverage': coverage
    }


def main():
    parser = argparse.ArgumentParser(
        description='Simplified SLAM - EKF-SLAM with simulated 2D LiDAR'
    )
    parser.add_argument('--steps', type=int, default=SIM_STEPS,
                       help='Number of simulation steps')
    parser.add_argument('--seed', type=int, default=SIM_SEED,
                       help='Random seed')
    parser.add_argument('--no-animate', action='store_true',
                       help='Disable real-time animation')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable all visualization')
    parser.add_argument('--save-gif', type=str, default=None,
                       help='Save animation as GIF')
    
    args = parser.parse_args()
    
    run_slam(
        num_steps=args.steps,
        seed=args.seed,
        visualize=not args.no_visualize,
        animate=not args.no_animate,
        save_animation=args.save_gif
    )


if __name__ == '__main__':
    main()
