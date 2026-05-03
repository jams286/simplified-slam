"""
Visualización animada del sistema SLAM.
Muestra el robot explorando y construyendo el mapa simultáneamente.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple

from .environment import Environment
from .occupancy_grid import OccupancyGrid


class SLAMVisualizer:
    """
    Visualizador en tiempo real del proceso SLAM.
    
    Panel izquierdo: entorno real con robot y escaneo LiDAR
    Panel derecho: mapa de ocupación estimado
    """

    def __init__(self, environment: Environment, 
                 occupancy_grid: OccupancyGrid,
                 figsize: Tuple[int, int] = (14, 6)):
        self.env = environment
        self.grid = occupancy_grid
        self.figsize = figsize
        
        # Datos acumulados para la animación
        self.frames_data: List[dict] = []

    def record_frame(self, true_pose: np.ndarray, 
                     estimated_pose: np.ndarray,
                     scan_endpoints: Optional[np.ndarray] = None,
                     landmarks_true: Optional[np.ndarray] = None,
                     landmarks_estimated: Optional[np.ndarray] = None,
                     true_path: Optional[List] = None,
                     estimated_path: Optional[List] = None,
                     pose_covariance: Optional[np.ndarray] = None):
        """Registra un frame para la animación."""
        self.frames_data.append({
            'true_pose': true_pose.copy(),
            'estimated_pose': estimated_pose.copy(),
            'scan_endpoints': scan_endpoints.copy() if scan_endpoints is not None else None,
            'landmarks_true': landmarks_true.copy() if landmarks_true is not None else None,
            'landmarks_estimated': landmarks_estimated.copy() if landmarks_estimated is not None else None,
            'true_path': list(true_path) if true_path else None,
            'estimated_path': list(estimated_path) if estimated_path else None,
            'pose_covariance': pose_covariance.copy() if pose_covariance is not None else None,
            'grid_snapshot': self.grid.get_probability_map().copy()
        })

    def animate(self, interval: int = 100, save_path: Optional[str] = None):
        """
        Genera la animación del proceso SLAM.
        
        Args:
            interval: milisegundos entre frames
            save_path: ruta para guardar como GIF (opcional)
        """
        fig, (ax_env, ax_map) = plt.subplots(1, 2, figsize=self.figsize)
        
        def init():
            ax_env.clear()
            ax_map.clear()
            return []
        
        def update(frame_idx):
            if frame_idx >= len(self.frames_data):
                return []
            
            data = self.frames_data[frame_idx]
            ax_env.clear()
            ax_map.clear()
            
            # === Panel izquierdo: Entorno real ===
            ax_env.set_xlim(-0.5, self.env.width + 0.5)
            ax_env.set_ylim(-0.5, self.env.height + 0.5)
            ax_env.set_aspect('equal')
            ax_env.set_title(f'Entorno Real (paso {frame_idx+1}/{len(self.frames_data)})')
            ax_env.set_xlabel('X (m)')
            ax_env.set_ylabel('Y (m)')
            
            # Dibujar paredes
            for wall in self.env.walls:
                ax_env.plot([wall.x1, wall.x2], [wall.y1, wall.y2], 
                          'k-', linewidth=2)
            
            # Dibujar escaneo LiDAR
            if data['scan_endpoints'] is not None:
                ax_env.scatter(data['scan_endpoints'][:, 0], 
                             data['scan_endpoints'][:, 1],
                             c='red', s=1, alpha=0.5, label='LiDAR')
            
            # Dibujar trayectoria real
            if data['true_path']:
                path = np.array(data['true_path'])
                ax_env.plot(path[:, 0], path[:, 1], 'b-', 
                          alpha=0.5, linewidth=1, label='Trayectoria real')
            
            # Dibujar trayectoria estimada
            if data['estimated_path']:
                epath = np.array(data['estimated_path'])
                ax_env.plot(epath[:, 0], epath[:, 1], 'g--', 
                          alpha=0.7, linewidth=1, label='Estimada')
            
            # Dibujar robot (pose real)
            tp = data['true_pose']
            robot_circle = plt.Circle((tp[0], tp[1]), 0.3, 
                                     color='blue', fill=False, linewidth=2)
            ax_env.add_patch(robot_circle)
            # Dirección
            ax_env.arrow(tp[0], tp[1], 
                        0.5*np.cos(tp[2]), 0.5*np.sin(tp[2]),
                        head_width=0.15, head_length=0.1, fc='blue', ec='blue')
            
            # Landmarks reales
            if data['landmarks_true'] is not None and len(data['landmarks_true']) > 0:
                ax_env.scatter(data['landmarks_true'][:, 0],
                             data['landmarks_true'][:, 1],
                             marker='^', c='orange', s=50, 
                             zorder=5, label='Landmarks reales')
            
            # Landmarks estimados
            if data['landmarks_estimated'] is not None and len(data['landmarks_estimated']) > 0:
                ax_env.scatter(data['landmarks_estimated'][:, 0],
                             data['landmarks_estimated'][:, 1],
                             marker='x', c='green', s=50, 
                             zorder=5, label='Landmarks estimados')
            
            ax_env.legend(loc='upper right', fontsize=7)
            
            # === Panel derecho: Mapa de ocupación ===
            ax_map.imshow(data['grid_snapshot'], origin='lower',
                        cmap='gray_r', vmin=0, vmax=1,
                        extent=[0, self.env.width, 0, self.env.height])
            ax_map.set_title('Mapa de Ocupación Estimado')
            ax_map.set_xlabel('X (m)')
            ax_map.set_ylabel('Y (m)')
            ax_map.set_aspect('equal')
            
            # Pose estimada en el mapa
            ep = data['estimated_pose']
            ax_map.plot(ep[0], ep[1], 'go', markersize=8)
            ax_map.arrow(ep[0], ep[1],
                        0.5*np.cos(ep[2]), 0.5*np.sin(ep[2]),
                        head_width=0.15, head_length=0.1, fc='green', ec='green')
            
            plt.tight_layout()
            return []
        
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=len(self.frames_data),
                           interval=interval, blit=False, repeat=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animación guardada en: {save_path}")
        
        plt.show()
        return anim

    def plot_final_comparison(self, true_path: List, estimated_path: List,
                              ground_truth_map: np.ndarray,
                              save_path: Optional[str] = None):
        """
        Genera una figura de comparación final con 4 paneles:
        1. Trayectorias (real vs estimada)
        2. Mapa ground truth
        3. Mapa estimado
        4. Diferencia entre mapas
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: Trayectorias
        ax = axes[0, 0]
        true_arr = np.array(true_path)
        est_arr = np.array(estimated_path)
        ax.plot(true_arr[:, 0], true_arr[:, 1], 'b-', label='Real', linewidth=1.5)
        ax.plot(est_arr[:, 0], est_arr[:, 1], 'r--', label='Estimada', linewidth=1.5)
        for wall in self.env.walls:
            ax.plot([wall.x1, wall.x2], [wall.y1, wall.y2], 'k-', linewidth=1)
        ax.set_title('Trayectorias: Real vs Estimada')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.set_aspect('equal')
        
        # Panel 2: Ground truth
        ax = axes[0, 1]
        ax.imshow(ground_truth_map, origin='lower', cmap='gray_r',
                 extent=[0, self.env.width, 0, self.env.height])
        ax.set_title('Mapa Ground Truth')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        
        # Panel 3: Mapa estimado
        ax = axes[1, 0]
        estimated_map = self.grid.get_probability_map()
        ax.imshow(estimated_map, origin='lower', cmap='gray_r',
                 extent=[0, self.env.width, 0, self.env.height])
        ax.set_title('Mapa de Ocupación Estimado')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        
        # Panel 4: Diferencia
        ax = axes[1, 1]
        binary_estimated = self.grid.get_binary_map(threshold=0.6)
        # Redimensionar ground truth si es necesario
        gt_resized = ground_truth_map[:binary_estimated.shape[0], 
                                      :binary_estimated.shape[1]]
        difference = np.abs(gt_resized - binary_estimated)
        ax.imshow(difference, origin='lower', cmap='hot',
                 extent=[0, self.env.width, 0, self.env.height])
        ax.set_title('Diferencia |GT - Estimado|')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparación guardada en: {save_path}")
        plt.show()
