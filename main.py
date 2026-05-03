"""
Punto de entrada principal del sistema SLAM simplificado.
Orquesta todos los módulos: entorno, robot, LiDAR, EKF-SLAM,
mapa de ocupación, visualización y análisis de error.
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
    Ejecuta la simulación SLAM completa.
    
    Args:
        num_steps: número de pasos de simulación
        seed: semilla para reproducibilidad
        visualize: si se muestra visualización final
        animate: si se genera animación en tiempo real
        save_animation: ruta para guardar la animación (GIF)
    """
    np.random.seed(seed)
    print("🤖 SLAM Simplificado - EKF-SLAM con LiDAR 2D")
    print("=" * 50)
    
    # === 1. Crear entorno ===
    print("\n📦 Creando entorno 2D...")
    env = create_default_environment()
    print(f"   Dimensiones: {env.width}x{env.height} m")
    print(f"   Paredes: {len(env.walls)}")
    print(f"   Landmarks: {len(env.landmarks)}")
    
    # === 2. Inicializar robot ===
    initial_pose = np.array([2.0, 2.0, 0.0])
    robot = Robot(
        x=initial_pose[0], y=initial_pose[1], theta=initial_pose[2],
        noise_v=MOTION_NOISE_V, noise_w=MOTION_NOISE_W
    )
    print(f"\n🤖 Robot inicializado en ({robot.x:.1f}, {robot.y:.1f}, {np.degrees(robot.theta):.0f}°)")
    
    # === 3. Inicializar LiDAR ===
    lidar = LiDAR(
        num_beams=LIDAR_NUM_BEAMS, max_range=LIDAR_MAX_RANGE,
        min_range=LIDAR_MIN_RANGE, fov=LIDAR_FOV,
        noise_std=LIDAR_NOISE_STD, miss_probability=LIDAR_MISS_PROB
    )
    print(f"📡 LiDAR: {LIDAR_NUM_BEAMS} rayos, rango máx {LIDAR_MAX_RANGE}m, ruido σ={LIDAR_NOISE_STD}m")
    
    # === 4. Inicializar EKF-SLAM ===
    ekf = EKFSLAM(
        initial_pose=initial_pose,
        range_noise=EKF_RANGE_NOISE,
        bearing_noise=EKF_BEARING_NOISE,
        association_threshold=EKF_ASSOCIATION_THRESHOLD
    )
    print(f"🧮 EKF-SLAM: R_noise={EKF_RANGE_NOISE}, B_noise={EKF_BEARING_NOISE}")
    
    # === 5. Inicializar mapa de ocupación ===
    occ_grid = OccupancyGrid(
        width=ENV_WIDTH, height=ENV_HEIGHT, resolution=GRID_RESOLUTION,
        log_odd_free=GRID_LOG_ODD_FREE, log_odd_occ=GRID_LOG_ODD_OCC,
        log_odd_prior=GRID_LOG_ODD_PRIOR, log_odd_max=GRID_LOG_ODD_MAX,
        log_odd_min=GRID_LOG_ODD_MIN
    )
    print(f"🗺️  Grilla de ocupación: {occ_grid.rows}x{occ_grid.cols} celdas ({GRID_RESOLUTION}m/celda)")
    
    # === 6. Inicializar visualizador ===
    vis = SLAMVisualizer(env, occ_grid, figsize=VIS_FIGSIZE)
    
    # === 7. Generar comandos de exploración ===
    commands = generate_exploration_commands(num_steps, DT, ENV_WIDTH, ENV_HEIGHT)
    print(f"\n🎯 Iniciando exploración: {num_steps} pasos ({num_steps*DT:.1f}s)")
    print("-" * 50)
    
    # === 8. Bucle principal de simulación ===
    landmarks_true = np.array([[lm.x, lm.y] for lm in env.landmarks])
    frame_skip = max(1, num_steps // 100)  # Limitar frames para rendimiento
    
    for step in range(num_steps):
        v, omega = commands[step]
        
        # Mover robot (con ruido)
        robot.move(v, omega, DT, add_noise=True)
        
        # Verificar colisión — si no es libre, revertir
        if not env.is_free(robot.x, robot.y, radius=0.3):
            robot.x = robot.true_path[-2][0] if len(robot.true_path) > 1 else robot.x
            robot.y = robot.true_path[-2][1] if len(robot.true_path) > 1 else robot.y
            robot.true_path[-1] = (robot.x, robot.y, robot.theta)
            continue
        
        # Escaneo LiDAR
        scan = lidar.scan(env, robot.x, robot.y, robot.theta)
        
        # Extraer landmarks del escaneo
        landmark_obs = lidar.extract_landmarks(scan)
        
        # Paso EKF-SLAM
        ekf.step(v, omega, DT, landmark_obs)
        
        # Actualizar mapa de ocupación con pose estimada
        estimated_pose = ekf.robot_pose
        occ_grid.update_from_pose(scan, estimated_pose)
        
        # Registrar frame para animación (cada N pasos)
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
        
        # Progreso
        if (step + 1) % (num_steps // 10) == 0:
            ate = np.linalg.norm(robot.pose[:2] - estimated_pose[:2])
            print(f"   Paso {step+1:4d}/{num_steps} | "
                  f"Landmarks: {ekf.num_landmarks:3d} | "
                  f"ATE: {ate:.4f}m | "
                  f"Cobertura: {np.mean(occ_grid.get_explored_mask()):.1%}")
    
    print("-" * 50)
    print("✅ Simulación completada")
    
    # === 9. Análisis de error ===
    print("\n📊 Calculando métricas de error...")
    
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
    
    # Imprimir reporte
    report = ErrorAnalysis.generate_report(
        traj_error, orient_error, map_metrics, lm_error, coverage
    )
    print(report)
    
    # === 10. Visualización ===
    if save_animation:
        os.makedirs(os.path.dirname(save_animation), exist_ok=True)

    if visualize:
        print("\n🎨 Generando visualización final...")
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
        print("🎬 Generando animación...")
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
        description='SLAM Simplificado - EKF-SLAM con LiDAR 2D simulado'
    )
    parser.add_argument('--steps', type=int, default=SIM_STEPS,
                       help='Número de pasos de simulación')
    parser.add_argument('--seed', type=int, default=SIM_SEED,
                       help='Semilla aleatoria')
    parser.add_argument('--no-animate', action='store_true',
                       help='Desactivar animación en tiempo real')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Desactivar toda visualización')
    parser.add_argument('--save-gif', type=str, default=None,
                       help='Guardar animación como GIF')
    
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
