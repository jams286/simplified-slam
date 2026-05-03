"""
Análisis de error entre el mapa real y el estimado por SLAM.
Métricas cuantitativas de rendimiento del sistema.
"""

import numpy as np
from typing import Dict, Tuple, List
from .occupancy_grid import OccupancyGrid


class ErrorAnalysis:
    """
    Calcula métricas de error para evaluar el rendimiento del SLAM.
    
    Métricas implementadas:
    - Error de pose (ATE - Absolute Trajectory Error)
    - Error de mapa (precisión, recall, F1, IoU)
    - Error de landmarks
    - RMSE de trayectoria
    """

    @staticmethod
    def absolute_trajectory_error(true_path: List[Tuple], 
                                   estimated_path: List[Tuple]) -> Dict:
        """
        Calcula el Absolute Trajectory Error (ATE).
        
        ATE = √(1/N · Σᵢ ||pᵢ_true - pᵢ_est||²)
        
        Returns:
            Diccionario con RMSE, media, max, min del error de posición
        """
        n = min(len(true_path), len(estimated_path))
        true_arr = np.array(true_path[:n])[:, :2]  # solo x, y
        est_arr = np.array(estimated_path[:n])[:, :2]
        
        errors = np.linalg.norm(true_arr - est_arr, axis=1)
        
        return {
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'mean': float(np.mean(errors)),
            'max': float(np.max(errors)),
            'min': float(np.min(errors)),
            'std': float(np.std(errors)),
            'errors': errors
        }

    @staticmethod
    def orientation_error(true_path: List[Tuple], 
                          estimated_path: List[Tuple]) -> Dict:
        """
        Calcula el error de orientación a lo largo de la trayectoria.
        
        Returns:
            Diccionario con estadísticas del error angular (radianes)
        """
        n = min(len(true_path), len(estimated_path))
        true_theta = np.array([p[2] for p in true_path[:n]])
        est_theta = np.array([p[2] for p in estimated_path[:n]])
        
        # Diferencia angular normalizada
        errors = np.abs(true_theta - est_theta)
        errors = np.minimum(errors, 2*np.pi - errors)
        
        return {
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'mean': float(np.mean(errors)),
            'max': float(np.max(errors)),
            'errors': errors
        }

    @staticmethod
    def map_accuracy(ground_truth: np.ndarray, 
                     estimated_map: np.ndarray,
                     explored_mask: np.ndarray = None) -> Dict:
        """
        Calcula métricas de precisión del mapa de ocupación.
        
        Solo evalúa celdas que han sido exploradas.
        
        Métricas:
        - Accuracy: (TP + TN) / Total
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        - F1: 2·P·R / (P + R)
        - IoU: TP / (TP + FP + FN)
        
        Returns:
            Diccionario con todas las métricas
        """
        # Asegurar mismas dimensiones
        min_rows = min(ground_truth.shape[0], estimated_map.shape[0])
        min_cols = min(ground_truth.shape[1], estimated_map.shape[1])
        
        gt = ground_truth[:min_rows, :min_cols]
        est = estimated_map[:min_rows, :min_cols]
        
        # Binarizar
        gt_binary = (gt > 0.5).astype(bool)
        est_binary = (est > 0.5).astype(bool)
        
        # Aplicar máscara de exploración si existe
        if explored_mask is not None:
            mask = explored_mask[:min_rows, :min_cols]
            gt_binary = gt_binary[mask]
            est_binary = est_binary[mask]
        
        gt_flat = gt_binary.flatten()
        est_flat = est_binary.flatten()
        
        tp = np.sum(gt_flat & est_flat)
        tn = np.sum(~gt_flat & ~est_flat)
        fp = np.sum(~gt_flat & est_flat)
        fn = np.sum(gt_flat & ~est_flat)
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou': float(iou),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_cells': int(total)
        }

    @staticmethod
    def landmark_error(true_landmarks: np.ndarray,
                       estimated_landmarks: np.ndarray) -> Dict:
        """
        Calcula el error de posición de landmarks estimados.
        
        Usa asociación por distancia mínima (greedy matching).
        
        Returns:
            Diccionario con estadísticas de error de landmarks
        """
        if len(true_landmarks) == 0 or len(estimated_landmarks) == 0:
            return {
                'mean_error': float('inf'),
                'num_matched': 0,
                'num_true': len(true_landmarks),
                'num_estimated': len(estimated_landmarks)
            }
        
        # Matching greedy por distancia mínima
        used = set()
        errors = []
        
        for est in estimated_landmarks:
            min_dist = float('inf')
            best_idx = -1
            for j, true in enumerate(true_landmarks):
                if j in used:
                    continue
                dist = np.linalg.norm(est - true)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = j
            
            if best_idx >= 0 and min_dist < 3.0:  # umbral de matching
                used.add(best_idx)
                errors.append(min_dist)
        
        if errors:
            errors = np.array(errors)
            return {
                'mean_error': float(np.mean(errors)),
                'rmse': float(np.sqrt(np.mean(errors**2))),
                'max_error': float(np.max(errors)),
                'num_matched': len(errors),
                'num_true': len(true_landmarks),
                'num_estimated': len(estimated_landmarks),
                'match_rate': len(errors) / len(true_landmarks)
            }
        
        return {
            'mean_error': float('inf'),
            'num_matched': 0,
            'num_true': len(true_landmarks),
            'num_estimated': len(estimated_landmarks),
            'match_rate': 0.0
        }

    @staticmethod
    def exploration_coverage(explored_mask: np.ndarray) -> Dict:
        """
        Calcula el porcentaje del mapa que ha sido explorado.
        """
        total = explored_mask.size
        explored = np.sum(explored_mask)
        return {
            'coverage_ratio': float(explored / total),
            'explored_cells': int(explored),
            'total_cells': int(total)
        }

    @staticmethod
    def generate_report(trajectory_error: Dict, orientation_error: Dict,
                        map_metrics: Dict, landmark_error: Dict,
                        coverage: Dict) -> str:
        """
        Genera un reporte textual completo del análisis de error.
        """
        report = []
        report.append("=" * 60)
        report.append("   REPORTE DE ANÁLISIS DE ERROR - SLAM SIMPLIFICADO")
        report.append("=" * 60)
        
        report.append("\n📍 ERROR DE TRAYECTORIA (ATE)")
        report.append(f"   RMSE:     {trajectory_error['rmse']:.4f} m")
        report.append(f"   Media:    {trajectory_error['mean']:.4f} m")
        report.append(f"   Máximo:   {trajectory_error['max']:.4f} m")
        report.append(f"   Std:      {trajectory_error['std']:.4f} m")
        
        report.append("\n🧭 ERROR DE ORIENTACIÓN")
        report.append(f"   RMSE:     {orientation_error['rmse']:.4f} rad ({np.degrees(orientation_error['rmse']):.2f}°)")
        report.append(f"   Media:    {orientation_error['mean']:.4f} rad ({np.degrees(orientation_error['mean']):.2f}°)")
        report.append(f"   Máximo:   {orientation_error['max']:.4f} rad ({np.degrees(orientation_error['max']):.2f}°)")
        
        report.append("\n🗺️  PRECISIÓN DEL MAPA")
        report.append(f"   Accuracy:  {map_metrics['accuracy']:.4f} ({map_metrics['accuracy']*100:.1f}%)")
        report.append(f"   Precision: {map_metrics['precision']:.4f}")
        report.append(f"   Recall:    {map_metrics['recall']:.4f}")
        report.append(f"   F1-Score:  {map_metrics['f1_score']:.4f}")
        report.append(f"   IoU:       {map_metrics['iou']:.4f}")
        
        report.append("\n📐 ERROR DE LANDMARKS")
        report.append(f"   Error medio:    {landmark_error.get('mean_error', 'N/A')}")
        report.append(f"   Landmarks det.: {landmark_error.get('num_estimated', 0)}")
        report.append(f"   Landmarks real: {landmark_error.get('num_true', 0)}")
        report.append(f"   Tasa match:     {landmark_error.get('match_rate', 0):.2%}")
        
        report.append("\n🔍 COBERTURA DE EXPLORACIÓN")
        report.append(f"   Cobertura: {coverage['coverage_ratio']:.2%}")
        report.append(f"   Celdas exploradas: {coverage['explored_cells']:,}")
        report.append(f"   Celdas totales:    {coverage['total_cells']:,}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
