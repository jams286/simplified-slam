# 🤖 Simplified SLAM — EKF-SLAM with 2D LiDAR

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![NumPy](https://img.shields.io/badge/NumPy-2.0+-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=flat&logo=plotly&logoColor=white)

- **EKF-SLAM** (Extended Kalman Filter) implemented from scratch without ROS dependencies
- 2D environment simulator with walls, rectangular obstacles, and corridors
- **2D LiDAR** scan generator with realistic Gaussian noise and missed readings
- **Occupancy grid map** updated in real-time using log-odds model
- Dual-panel animated visualization: real environment vs estimated map
- Quantitative error analysis (ATE, map precision, IoU, landmark error)
- **70 tests** with **100% pass rate**

> Built as a portfolio project to demonstrate understanding of probabilistic robotics, state estimation, and simultaneous localization and mapping from first principles.

---

## 📐 Mathematical Foundation

### Motion Model (Velocity Motion Model)

```
x' = x + v·cos(θ)·dt
y' = y + v·sin(θ)·dt
θ' = θ + ω·dt
```

| Symbol | Description | Unit |
|--------|-------------|------|
| `x, y` | Robot position | m |
| `θ` | Orientation | rad |
| `v` | Linear velocity | m/s |
| `ω` | Angular velocity | rad/s |
| `dt` | Time step | s |

### EKF-SLAM — Augmented State

The state vector contains the robot pose and all observed landmark positions:

```
μ = [x, y, θ, lx₁, ly₁, lx₂, ly₂, ..., lxₙ, lyₙ]
Σ ∈ ℝ^(3+2N)×(3+2N)
```

**Prediction:**
```
μ̄ = g(μₜ₋₁, uₜ)
Σ̄ = Gₜ·Σₜ₋₁·Gₜᵀ + Rₜ
```

**Correction (for each observation zᵢ = [r, φ]):**
```
Expected measurement:  ẑ = h(μ̄) = [√((lx-x)² + (ly-y)²), atan2(ly-y, lx-x) - θ]
Innovation:            ν = z - ẑ
Kalman Gain:           K = Σ̄·Hᵀ·(H·Σ̄·Hᵀ + Q)⁻¹
Update:                μ = μ̄ + K·ν
                       Σ = (I - K·H)·Σ̄
```

| Symbol | Description |
|--------|-------------|
| `μ` | State vector (pose + landmarks) |
| `Σ` | Joint covariance matrix |
| `G` | Motion model Jacobian |
| `H` | Observation model Jacobian |
| `K` | Kalman Gain |
| `R` | Process noise covariance |
| `Q` | Measurement noise covariance |
| `ν` | Innovation (residual) |

### Occupancy Grid — Log-Odds

```
L(m|z₁:ₜ) = L(m|z₁:ₜ₋₁) + L(m|zₜ) - L₀

where L = log(p / (1-p))
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `L_free` | -0.4 | Log-odds for free space |
| `L_occ` | 0.9 | Log-odds for occupied cell |
| `L₀` | 0.0 | Prior (maximum uncertainty) |
| `L_max` | 5.0 | Maximum saturation |
| `L_min` | -5.0 | Minimum saturation |

### Ray-Casting Algorithm (Bresenham)

LiDAR scans are projected onto the map using Bresenham's line algorithm to efficiently trace each ray in the discrete grid, marking cells as free (traversed) or occupied (endpoint).

---

## 🗂️ Project Structure

```
simplified-slam/
├── main.py                     # Simulation mode — full SLAM with simulated LiDAR
├── main_real.py                # Real LiDAR mode — live SLAM with CSPC sensor
├── config.py                   # Global configurable parameters
├── run_tests.py                # Full test suite runner
├── requirements.txt            # Dependencies: numpy, matplotlib, pyserial
├── LICENSE                     # MIT License
├── .gitignore                  # Files excluded from git
├── drivers/
│   └── cspc_lidar/             # CSPC M1C1/M1CT LiDAR driver (serial protocol)
│       ├── cspc_lidar.py       # Core driver — serial communication & packet parsing
│       ├── visualizer.py       # Standalone polar/cartesian scan viewer
│       ├── record_csv.py       # Record scans to CSV file
│       └── merge_scans.py      # Merge multiple scan sessions
├── src/
│   ├── __init__.py
│   ├── environment.py          # 2D environment simulator (walls, obstacles, ray-casting)
│   ├── robot.py                # Differential drive robot kinematic model
│   ├── lidar.py                # Simulated 2D LiDAR sensor with Gaussian noise
│   ├── real_lidar.py           # Real LiDAR adapter (CSPC → SLAM format)
│   ├── scan_matching.py        # ICP scan matcher (motion estimation without odometry)
│   ├── ekf_slam.py             # EKF-SLAM algorithm (predict-update with landmarks)
│   ├── occupancy_grid.py       # Occupancy grid map (log-odds + Bresenham)
│   ├── visualization.py        # Dual-panel animated visualization
│   └── error_analysis.py       # Error metrics (ATE, F1, IoU, landmarks)
└── tests/
    ├── __init__.py
    ├── test_environment.py     # 14 tests — ray-casting, collisions, ground truth
    ├── test_lidar.py           # 13 tests — scans, noise, landmarks
    ├── test_ekf_slam.py        # 14 tests — prediction, correction, data association
    ├── test_occupancy_grid.py  # 14 tests — log-odds, Bresenham, updates
    └── test_error_analysis.py  # 15 tests — ATE, map metrics, coverage
```

---

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/jams286/simplified-slam.git
   cd simplified-slam
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the simulation (no hardware needed)**
   ```bash
   python main.py
   ```

5. **Run with real LiDAR (CSPC M1C1/M1CT)**
   ```bash
   python main_real.py --port COM3 --version 2
   ```

6. **Run tests**
   ```bash
   python run_tests.py
   ```

---

## 🎮 Usage

### Command Line

```bash
# Default simulation (500 steps, with animation)
python main.py

# Long simulation without animation
python main.py --steps 1000 --no-animate

# Metrics only (no visualization)
python main.py --no-visualize

# Save animation as GIF
python main.py --save-gif results/slam_demo.gif

# Change random seed
python main.py --seed 123
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--steps` | 500 | Number of simulation steps |
| `--seed` | 42 | Random seed for reproducibility |
| `--no-animate` | False | Disable real-time animation |
| `--no-visualize` | False | Disable all visualization |
| `--save-gif` | None | Path to save animation GIF |

### Real LiDAR Mode

```bash
# Basic usage (plug in LiDAR, find port in Device Manager)
python main_real.py --port COM3 --version 2

# With custom baud rate and range
python main_real.py --port COM3 --version 2 --baud 230400 --max-range 5.0

# Smaller map for a single room
python main_real.py --port COM3 --version 2 --grid-size 10

# Linux
python main_real.py --port /dev/ttyUSB0 --version 3
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--port` | COM3 | Serial port |
| `--version` | 2 | LiDAR model (1=M1C1_v1, 2=M1C1_v2, 3=Coin_Plus, 4=Coin_D2) |
| `--baud` | auto | Baud rate (auto-detected from version) |
| `--max-range` | 8.0 | Maximum range in meters |
| `--min-range` | 0.10 | Minimum range in meters |
| `--grid-size` | 20.0 | Map size in meters (square) |
| `--grid-resolution` | 0.1 | Grid cell size in meters |
| `--max-steps` | 0 | Max scans (0 = unlimited, Ctrl+C to stop) |

**How it works (handheld mode):**
1. The LiDAR spins and captures 360° scans
2. ICP scan matching compares consecutive scans to estimate your movement
3. EKF-SLAM fuses the motion estimate with landmark observations
4. The occupancy grid builds the map in real-time
5. Walk around the room — the map grows as you explore!

### Configurable Parameters (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ENV_WIDTH/HEIGHT` | 20.0 m | Environment dimensions |
| `LIDAR_NUM_BEAMS` | 180 | LiDAR ray count |
| `LIDAR_MAX_RANGE` | 8.0 m | Maximum sensor range |
| `LIDAR_NOISE_STD` | 0.05 m | Gaussian noise σ |
| `GRID_RESOLUTION` | 0.1 m | Occupancy map resolution |
| `EKF_ASSOCIATION_THRESHOLD` | 1.5 | Data association threshold |

---

## 📊 Test Results

**70 tests passed — 100% pass rate**

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestEnvironment` | 14 | Ray-casting, collisions, boundary walls, ground truth map |
| `TestLiDAR` | 12 | Scans, Gaussian noise, missed readings, landmark extraction |
| `TestLiDARScan` | 1 | Valid measurement count |
| `TestEKFSLAM` | 14 | Prediction, correction, data association, convergence, covariance |
| `TestOccupancyGrid` | 14 | Log-odds, Bresenham, saturation, coordinate conversion |
| `TestErrorAnalysis` | 15 | ATE, orientation, precision/recall/F1/IoU, landmarks, coverage |

---

## 📊 Metrics / Results

Measured results with default configuration (500 steps, seed=42):

| Metric | Value | Description |
|--------|-------|-------------|
| **ATE RMSE** | ~0.15 m | Absolute Trajectory Error |
| **Orientation RMSE** | ~0.08 rad (~4.6°) | Heading estimation error |
| **Map Accuracy** | >90% | Occupancy grid accuracy |
| **Map F1-Score** | >0.6 | Precision-recall balance |
| **Exploration Coverage** | >40% | Percentage of map explored |
| **Landmark Match Rate** | >60% | Correct landmark association rate |

*Note: Exact values vary depending on the exploration trajectory and random seed.*

---

## 🗺️ Roadmap

- [x] 2D environment simulator with obstacles
- [x] LiDAR scan generator with Gaussian noise
- [x] EKF-SLAM with point landmarks
- [x] Occupancy grid map (log-odds)
- [x] Dual-panel animated visualization
- [x] Quantitative error analysis
- [x] Full test suite (70 tests)
- [ ] Autonomous exploration with frontier-based exploration
- [ ] Alternative implementation with FastSLAM (Particle Filter)
- [ ] Loop closure detection
- [x] Support for real CSPC LiDAR (M1C1 / M1CT) with scan matching
- [ ] Map export to PGM/YAML format
- [ ] Interactive web interface with real-time controls

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
