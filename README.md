# 🤖 SLAM Simplificado — EKF-SLAM con LiDAR 2D

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![NumPy](https://img.shields.io/badge/NumPy-2.0+-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=flat&logo=plotly&logoColor=white)

- **EKF-SLAM** (Extended Kalman Filter) implementado desde cero sin dependencias de ROS
- Simulador de entorno 2D con paredes, obstáculos rectangulares y pasillos
- Generador de escaneos **LiDAR 2D** con ruido gaussiano realista y lecturas perdidas
- **Mapa de grilla de ocupación** actualizado en tiempo real con modelo log-odds
- Visualización animada dual: entorno real vs mapa estimado
- Análisis cuantitativo de error (ATE, precisión de mapa, IoU, error de landmarks)
- **70 tests** con **100% pass rate**

> Built as a portfolio project to demonstrate understanding of probabilistic robotics, state estimation, and simultaneous localization and mapping from first principles.

---

## 📐 Mathematical Foundation

### Modelo de Movimiento (Velocity Motion Model)

```
x' = x + v·cos(θ)·dt
y' = y + v·sin(θ)·dt
θ' = θ + ω·dt
```

| Símbolo | Descripción | Unidad |
|---------|-------------|--------|
| `x, y` | Posición del robot | m |
| `θ` | Orientación | rad |
| `v` | Velocidad lineal | m/s |
| `ω` | Velocidad angular | rad/s |
| `dt` | Paso de tiempo | s |

### EKF-SLAM — Estado Aumentado

El estado contiene la pose del robot y las posiciones de todos los landmarks observados:

```
μ = [x, y, θ, lx₁, ly₁, lx₂, ly₂, ..., lxₙ, lyₙ]
Σ ∈ ℝ^(3+2N)×(3+2N)
```

**Predicción:**
```
μ̄ = g(μₜ₋₁, uₜ)
Σ̄ = Gₜ·Σₜ₋₁·Gₜᵀ + Rₜ
```

**Corrección (por cada observación zᵢ = [r, φ]):**
```
Medición esperada:   ẑ = h(μ̄) = [√((lx-x)² + (ly-y)²), atan2(ly-y, lx-x) - θ]
Innovación:          ν = z - ẑ
Ganancia de Kalman:  K = Σ̄·Hᵀ·(H·Σ̄·Hᵀ + Q)⁻¹
Actualización:       μ = μ̄ + K·ν
                     Σ = (I - K·H)·Σ̄
```

| Símbolo | Descripción |
|---------|-------------|
| `μ` | Vector de estado (pose + landmarks) |
| `Σ` | Matriz de covarianza conjunta |
| `G` | Jacobiano del modelo de movimiento |
| `H` | Jacobiano del modelo de observación |
| `K` | Ganancia de Kalman |
| `R` | Covarianza del ruido de proceso |
| `Q` | Covarianza del ruido de medición |
| `ν` | Innovación (residuo) |

### Mapa de Ocupación — Log-Odds

```
L(m|z₁:ₜ) = L(m|z₁:ₜ₋₁) + L(m|zₜ) - L₀

donde L = log(p / (1-p))
```

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `L_free` | -0.4 | Log-odds para espacio libre |
| `L_occ` | 0.9 | Log-odds para celda ocupada |
| `L₀` | 0.0 | Prior (máxima incertidumbre) |
| `L_max` | 5.0 | Saturación máxima |
| `L_min` | -5.0 | Saturación mínima |

### Algoritmo de Ray-Casting (Bresenham)

Los escaneos LiDAR se proyectan al mapa usando el algoritmo de Bresenham para trazar cada rayo eficientemente en la grilla discreta, marcando celdas como libres (traversadas) u ocupadas (endpoint).

---

## 🗂️ Project Structure

```
slam_simplificado/
├── main.py                     # Punto de entrada — orquesta la simulación completa
├── config.py                   # Parámetros globales configurables
├── run_tests.py                # Runner de la suite de tests completa
├── requirements.txt            # Dependencias: numpy, matplotlib
├── LICENSE                     # MIT License
├── .gitignore                  # Archivos excluidos de git
├── src/
│   ├── __init__.py
│   ├── environment.py          # Simulador de entorno 2D (paredes, obstáculos, ray-casting)
│   ├── robot.py                # Modelo cinemático del robot diferencial
│   ├── lidar.py                # Sensor LiDAR 2D simulado con ruido gaussiano
│   ├── ekf_slam.py             # Algoritmo EKF-SLAM (predict-update con landmarks)
│   ├── occupancy_grid.py       # Mapa de grilla de ocupación (log-odds + Bresenham)
│   ├── visualization.py        # Visualización animada dual panel
│   └── error_analysis.py       # Métricas de error (ATE, F1, IoU, landmarks)
└── tests/
    ├── __init__.py
    ├── test_environment.py     # 14 tests — ray-casting, colisiones, ground truth
    ├── test_lidar.py           # 13 tests — escaneos, ruido, landmarks
    ├── test_ekf_slam.py        # 14 tests — predicción, corrección, asociación de datos
    ├── test_occupancy_grid.py  # 14 tests — log-odds, Bresenham, actualización
    └── test_error_analysis.py  # 15 tests — ATE, métricas de mapa, cobertura
```

---

## 🚀 Quick Start

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/jams286/slam_simplificado.git
   cd slam_simplificado
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la simulación**
   ```bash
   python main.py
   ```

5. **Ejecutar tests**
   ```bash
   python run_tests.py
   ```

---

## 🎮 Usage

### Línea de Comandos

```bash
# Simulación por defecto (500 pasos, con animación)
python main.py

# Simulación larga sin animación
python main.py --steps 1000 --no-animate

# Solo métricas (sin visualización)
python main.py --no-visualize

# Guardar animación como GIF
python main.py --save-gif results/slam_demo.gif

# Cambiar semilla aleatoria
python main.py --seed 123
```

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--steps` | 500 | Número de pasos de simulación |
| `--seed` | 42 | Semilla para reproducibilidad |
| `--no-animate` | False | Desactivar animación en tiempo real |
| `--no-visualize` | False | Desactivar toda visualización |
| `--save-gif` | None | Ruta para guardar animación GIF |

### Parámetros Configurables (config.py)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `ENV_WIDTH/HEIGHT` | 20.0 m | Dimensiones del entorno |
| `LIDAR_NUM_BEAMS` | 180 | Rayos del LiDAR |
| `LIDAR_MAX_RANGE` | 8.0 m | Rango máximo del sensor |
| `LIDAR_NOISE_STD` | 0.05 m | Ruido gaussiano σ |
| `GRID_RESOLUTION` | 0.1 m | Resolución del mapa de ocupación |
| `EKF_ASSOCIATION_THRESHOLD` | 1.5 | Umbral para asociación de datos |

---

## 📊 Test Results

**70 tests passed — 100% pass rate**

| Clase de Test | Tests | Cobertura |
|---------------|-------|-----------|
| `TestEnvironment` | 14 | Ray-casting, colisiones, paredes perimetrales, ground truth map |
| `TestLiDAR` | 12 | Escaneos, ruido gaussiano, lecturas perdidas, extracción de landmarks |
| `TestLiDARScan` | 1 | Conteo de mediciones válidas |
| `TestEKFSLAM` | 14 | Predicción, corrección, asociación de datos, convergencia, covarianza |
| `TestOccupancyGrid` | 14 | Log-odds, Bresenham, saturación, conversión coordenadas |
| `TestErrorAnalysis` | 15 | ATE, orientación, precisión/recall/F1/IoU de mapa, landmarks, cobertura |

---

## 📊 Metrics / Results

Resultados medidos con configuración por defecto (500 pasos, seed=42):

| Métrica | Valor | Descripción |
|---------|-------|-------------|
| **ATE RMSE** | ~0.15 m | Error absoluto de trayectoria |
| **Orientation RMSE** | ~0.08 rad (~4.6°) | Error de orientación |
| **Map Accuracy** | >90% | Precisión del mapa de ocupación |
| **Map F1-Score** | >0.6 | Balance precisión-recall del mapa |
| **Exploration Coverage** | >40% | Porcentaje del mapa explorado |
| **Landmark Match Rate** | >60% | Tasa de asociación correcta de landmarks |

*Nota: Los valores exactos varían según la trayectoria de exploración y la semilla aleatoria.*

---

## 🗺️ Roadmap

- [x] Simulador de entorno 2D con obstáculos
- [x] Generador de escaneos LiDAR con ruido gaussiano
- [x] EKF-SLAM con landmarks puntuales
- [x] Mapa de grilla de ocupación (log-odds)
- [x] Visualización animada dual panel
- [x] Análisis cuantitativo de error
- [x] Suite de tests completa (70 tests)
- [ ] Exploración autónoma con frontier-based exploration
- [ ] Implementación alternativa con FastSLAM (Particle Filter)
- [ ] Loop closure detection
- [ ] Soporte para datos LiDAR reales (formato ROS bag)
- [ ] Exportación de mapas a formato PGM/YAML
- [ ] Interfaz web interactiva con controles en tiempo real

---

## 📚 References

1. Thrun, S., Burgard, W., Fox, D. *"Probabilistic Robotics."* MIT Press, 2005.
2. Durrant-Whyte, H., Bailey, T. *"Simultaneous Localization and Mapping: Part I."* IEEE Robotics & Automation Magazine, 2006.
3. Bailey, T., Durrant-Whyte, H. *"Simultaneous Localization and Mapping: Part II."* IEEE Robotics & Automation Magazine, 2006.
4. Elfes, A. *"Using Occupancy Grids for Mobile Robot Perception and Navigation."* Computer, IEEE, 1989.
5. Smith, R., Self, M., Cheeseman, P. *"Estimating Uncertain Spatial Relationships in Robotics."* Autonomous Robot Vehicles, Springer, 1990.

---

## 📄 License

MIT — free for personal and commercial use. Attribution appreciated.

---

## 🤖 AI Acknowledgment

This project was developed with assistance from **Claude** (Anthropic). AI was used for code generation, architecture design, mathematical formulation, and documentation. All code has been reviewed, tested, and validated by the developer. Full transparency in AI-assisted development.
