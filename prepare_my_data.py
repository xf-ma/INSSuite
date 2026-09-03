
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import json
from pathlib import Path
import scipy.io as sio
import glob
import os
import numpy as np
import xp
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Tuple, List, Union
from sins import IMU
import copy

@dataclass(slots=True)
class OptimizerConfig:
    # User/application specific fields:
    initial_guess: Optional[Any] = None

    # Optimization related fields:
    optimizer_type: str = 'AD'  # Options: 'AD', 'ABC', 'SCIPY', 'GRIDSEARCH'
    optimize_method: str = 'ADAM'  # Options: LBFGS, ADAM, SGD for 'AD'; L-BFGS-B, Nelder-Mead for 'SCIPY'; ignored for 'ABC', 'GRIDSEARCH'
    optimize_iterations: int = 50
    optimize_step: float = 1.0 # L-BFGS typically uses larger lr # learning_rate=3e-3  # Adam smaller lr
    patience: int = 110
    loss_threshold: float = 0.001
    optimize_bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(1.0, 30.0)])
    num_grids: Tuple[int, int] = (10, 10)

    prior: List[Tuple[float, float]] = field(default_factory=lambda: [(1.0, 30.0)])
    max_populations: int = 8
    n_samples: int = 10

    # Visualization fields:
    idx_visualize: tuple = field(default_factory=lambda: (0,))
    ifplot: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict):
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        
        for key, value in config_dict.items():
            if key in valid_fields:
                filtered[key] = value
            else:
                print(f"Warning: Unknown config key '{key}' will be ignored")
        
        return cls(**filtered)
    
    def update(self, updates: dict):
        valid_fields = {f.name for f in fields(self)}
        for key, value in updates.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} does not exist in GlobalConfig")
    
@dataclass(slots=True)
class GlobalConfig:
    imu_path: str = None
    truth_path: str = None

    imu_folder: str = None
    truth_folder: str = None
    trail_id: Optional[str] = None

    special_id: Optional[Any] = None

    fs: float = None

    ref_grid_path: str = None
    grid_scale: float = 1

    truth: Optional[Any] = None
    truth_sparse: Optional[Any] = None 

    cut_start: int = None
    cut_end: int = None
    still_time: int = 20

    SSM_name: str = "hybvio" # "HYBVIO", "SOLA", or "OPENSHOE"
    ZVDtype:  str = 'SHOE'  
    zupt_threshold:  float = 18.0
    min_zupt_duration_s:  float = 0.1
    use_reference_zupt: bool = False
    metric_type: str = "FRECHET"

    nav_mode: bool = True
    plot: bool = True

    params_user: Optional[Any] = None

    axis2D: Tuple[int, int] = (0, 1)

    optimize: Optional[OptimizerConfig] = None

    matern_kwargs: Optional[dict] = None     # extra keyword args forwarded to SSM_Matern.__init__
    sola_kwargs: Optional[dict] = None       # extra keyword args forwarded to SSM_Sola.__init__
    openshoe_kwargs: Optional[dict] = None   # extra keyword args forwarded to SSM_OpenShoe.__init__
    hybvio_kwargs: Optional[dict] = None     # extra keyword args forwarded to SSM_HybVIO.__init__

    def update(self, updates: dict):
        valid_fields = {f.name for f in fields(self)}
        for key, value in updates.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} does not exist in GlobalConfig")
    
    def create_optimizer_config(self, optimize_dict: dict = None):
        self.optimize = OptimizerConfig()
        self.optimize.update(optimize_dict)

def _read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def _trim_samples(data, cut_start, cut_end):
    n = data.shape[-1]

    if cut_start is None:
        cut_start = 0
    if cut_end is None:
        cut_end = 0
    
    i0 = max(0, int(cut_start))
    i1 = n - max(0, int(cut_end))
    if i1 <= i0:
        raise ValueError(f"Invalid trim [{i0}, {cut_end}] for length {n}.")
    return data[..., i0:i1]


def load_imu_data(config: GlobalConfig):
    # File path object
    if config.imu_path is not None:
        imu_file = Path(config.imu_path)
    else:
        # imu_trail_file = glob.glob(os.path.join(str(config.imu_folder), f"*{config.trail_id:02d}*.mat"))
        # if not imu_trail_file:
        #     print(f"\033[31m[ERROR] Found no file including {config.trail_id:02d}.\033[0m")
        # elif len(imu_trail_file)>1:
        #     print(f"\033[33m[WARNING] Found multiple files including {config.trail_id:02d}, select the first by default.\033[0m")
        # else:
        #     pass

        root = Path(str(config.imu_folder))
        trail_id = config.trail_id
        if trail_id is None:
            json_files = sorted(root.glob("*.json"))
            if len(json_files) != 1:
                raise FileNotFoundError(
                    "Ambiguous input: set config['imu_file'] explicitly or provide trail_id."
                )
            imu_file = json_files[0]
        else:
            cand = sorted(root.glob(f"*{trail_id}*.json"))
            if len(cand) == 0:
                raise FileNotFoundError(f"Found no file including trail_id={trail_id}")
            if len(cand) > 1:
                print(cand)
                print(
                    f"\033[33m[WARNING] Found multiple files including {trail_id}, "
                    "selecting the first by default.\033[0m"
                )
            imu_file = cand[0]

    # Load file
    if imu_file.suffix.lower() == ".json":
        imu_json = _read_json(imu_file)
        print(f"Loaded json file {str(imu_file)}")
        imu = IMU()
        imu.from_dict(imu_json)
    else:
        raise ValueError(f"Unsupported IMU file extension yet: {imu_file.suffix}")
    
    imu.source_file = imu_file
    if config.cut_start is not None and config.cut_end is not None:
        imu.trim(config.cut_start, config.cut_end)
    
    if config.special_id is not None:
        # DO SOMETHING
        pass
    
    return imu, imu_file
    

def load_truth_data(config: GlobalConfig, imu_fs: float):
    if config.truth_path is None and config.truth_folder is None and config.ref_grid_path is None:
        print(f"\033[33m[WARNING] No truth provided. Trajectory metric will be invalid.\033[0m")

        return np.array([[0],[0],[0]]), np.array([[0],[0],[0]]), None

    # ------  Grid ------
    if config.ref_grid_path is not None:
        if isinstance(config.ref_grid_path, (list, tuple, np.ndarray)):
            return np.asarray(config.ref_grid_path*config.grid_scale), np.array(config.ref_grid_path*config.grid_scale)#, None
        elif isinstance(config.ref_grid_path, str):
            grid_path = Path(config.ref_grid_path)
        else:
            return np.array([[0],[0],[0]]), np.array([[0],[0],[0]])#, None

        if grid_path.suffix.lower() == ".json":
            grid_data = _read_json(grid_path)
            key = "pos"
        else:
            raise ValueError(f"Unsupported grid file extension: {grid_path.suffix}")
        
        truth_sparse = np.asarray(grid_data[key]).T
        truth_sparse = truth_sparse * config.grid_scale
        truth = densify_curve(truth_sparse, samples_per_segment=150)

        return truth, truth_sparse#, grid_data
    
    # ------  Other ------
    if config.truth_path is not None:
        # DO SOMETHING
        pass
    

    return xp.asarray(truth), xp.asarray(truth_sparse)

def plot_truth(truth, truth_sparse=None, axis2D:Tuple[int, int]=(0, 1)):
    if truth is None:
        return
        
    truth_np = np.asarray(truth)
    if truth_np.shape[0] > truth_np.shape[1]:
        truth_np = truth_np.T
    if truth_np.shape[0] == 2:
        truth_np = np.vstack([truth_np, np.zeros((1, truth_np.shape[1]))])
    elif truth_np.shape[0] != 3:
        raise ValueError("truth must be 2D/3D trajectory with shape (3, N) or (N, 3)")

    fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{'type': 'xy'}, {'type': 'scene'}]],
            subplot_titles=("2D trajectory","3D trajectory"),
        )
    fig.add_trace(
        go.Scatter(
            x=truth_np[axis2D[0], :],
            y=truth_np[axis2D[1], :],
            mode="lines",
            name="Truth", legendgroup="Truth", showlegend=False,
            line=dict(color="#2e782c",width=2)
        ),
        row=1,col=1,
    )
    if truth_sparse is not None:
        fig.add_trace(
            go.Scatter(
                x=truth_sparse[axis2D[0], :],
                y=truth_sparse[axis2D[1], :],
                mode="markers",
                name="True static points", legendgroup="True static points", showlegend=False,
                marker=dict(color="#2e782c",size=7)
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter3d(
                x=truth_np[0, :], y=truth_np[1, :], z=truth_np[2, :],
                name="Truth", legendgroup="Truth", showlegend=True,
                mode="lines",
                line=dict(color="#2e782c",width=2)
            ),
        row=1,col=2,
    )
    if truth_sparse is not None:
        fig.add_trace(
            go.Scatter3d(
                x=truth_sparse[0, :],
                y=truth_sparse[1, :],
                z=truth_sparse[2, :],
                mode="markers",
                name="True static points", legendgroup="True static points", showlegend=True,
                marker=dict(color="#2e782c",size=7)
            ),
            row=1, col=2,
        )

    axis_titles = ['x [m]', 'y [m]', 'z [m]']
    fig.update_layout(
        title="Truth Trajectory with Static Interval Means",
        xaxis_title=axis_titles[axis2D[0]],
        yaxis_title=axis_titles[axis2D[1]],
        scene=dict(aspectmode="data")
    )

    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(row=1, col=1)

    fig.show()


def calibrate_my_imu(
    imu,
    still: int = 20,
    g: float = 9.80665):
    """
    Calibrate IMU measurements
    
    Args:
        imu: 6 or 9 x N array containing:
            - Accelerometer data (3 x N array)
            - Gyroscope data (3 x N array)
    """
    # Create a copy to avoid modifying original
    imu_calibrated = xp.copy(imu)
    if xp.asarray(imu_calibrated.acc).shape[0] > xp.asarray(imu_calibrated.acc).shape[1]:
        imu_calibrated.acc = xp.asarray(imu_calibrated.acc).T
        imu_calibrated.gyr = xp.asarray(imu_calibrated.gyr).T
        imu_calibrated.mag = xp.asarray(imu_calibrated.mag).T

    # Initialize bias_gyros
    bias_gyros = xp.zeros(3)

    # ======= Accelerometer Calibration =======
    if imu.whichimu == 'A':
        scale_acc = xp.eye(3)
        bias_acc = xp.array([[-0.0167], [0.0390], [-0.0094]])
        # Apply calibration: acc_calibrated = inv(scale) * (acc_raw - bias)
        acc_shifted = xp.asarray(imu_calibrated.acc.copy()) - xp.asarray(bias_acc)
        imu_calibrated.acc = xp.solve(xp.to_backend(scale_acc), acc_shifted)
    elif imu.whichimu == 'B':
        # OT calibration
        scale_acc = xp.eye(3)
        bias_acc = xp.array([[0.0717], [-0.0239], [-0.1582]])
        # Apply calibration: acc_calibrated = inv(scale) * (acc_raw - bias)
        acc_shifted = xp.asarray(imu_calibrated.acc.copy()) - xp.asarray(bias_acc)
        imu_calibrated.acc = xp.solve(xp.to_backend(scale_acc), acc_shifted)
      
    elif imu.whichimu=="A14A":
        bias_vec = xp.array([-0.0323791,   0.00117768, -0.05146826])
        calib_mtx = xp.array([[ 1.01623948e-01,  2.57668159e-05, -1.27672932e-04],
                            [ 2.57668159e-05,  1.01698165e-01, -1.75841192e-04],
                            [-1.27672932e-04, -1.75841192e-04,  1.01555796e-01]])
        acc_shifted = xp.asarray(imu_calibrated.acc.copy()) - xp.asarray(bias_vec).reshape(3,1)
        X_unit = calib_mtx @ acc_shifted   # norm ~ 1 (in g-units)
        imu_calibrated.acc = X_unit * g     

    elif imu.whichimu=="A14SF":
        bias_vec = xp.array([-0.01863516,  0.00123146, -0.03385071])
        calib_mtx = xp.array([[ 1.01779969e-01,  1.06251034e-04, -7.26862711e-05],
                            [ 1.06251034e-04,  1.01641757e-01, -1.72307622e-04],
                            [-7.26862711e-05, -1.72307622e-04,  1.01585593e-01]])
        acc_shifted = xp.asarray(imu_calibrated.acc.copy()) - xp.asarray(bias_vec).reshape(3,1)
        X_unit = calib_mtx @ acc_shifted   # norm ~ 1 (in g-units)
        imu_calibrated.acc = X_unit * g  
        
    elif imu.whichimu=="Galaxy" or imu.whichimu=="GalaxyS8":
        bias_vec = xp.array([-0.0210695,   0.00257689,  0.05762988])
        calib_mtx = xp.array([[ 1.02030969e-01,  1.57249097e-04, -5.01588238e-04],
                            [ 1.57249097e-04,  1.01169338e-01, -5.20113057e-05],
                            [-5.01588238e-04, -5.20113057e-05,  1.02012403e-01]])
        acc_shifted = xp.asarray(imu_calibrated.acc.copy()) - xp.asarray(bias_vec).reshape(3,1)
        X_unit = calib_mtx @ acc_shifted   # norm ~ 1 (in g-units)
        imu_calibrated.acc = X_unit * g  

    elif imu.whichimu=="RealmeX7pro" or imu.whichimu=="Realme":
        bias_vec = xp.array([0.2314887,  -0.13524598,  0.01894217])
        calib_mtx = xp.array([[0.10215338,  0.00015062, -0.00011444],
                            [ 0.00015062,  0.10195419, -0.00060707],
                            [-0.00011444, -0.00060707,  0.10080693]])
        acc_shifted = xp.asarray(imu_calibrated.acc.copy()) - xp.asarray(bias_vec).reshape(3,1)
        X_unit = calib_mtx @ acc_shifted   # norm ~ 1 (in g-units)
        imu_calibrated.acc = X_unit * g  
        
    else:
        raise ValueError(f"Unknown IMU identifier: {imu.whichimu}")
            
    
    # ======= Gyroscope Calibration =======
    # print(f"Compute gyro bias from static period ({still} samples)")
    bias_gyros = xp.mean(xp.asarray(imu_calibrated.gyr[:,:still]), axis=1, keepdims=True)
    imu_calibrated.gyr = xp.asarray(imu_calibrated.gyr) - bias_gyros.reshape(3, 1)

    return imu_calibrated, bias_gyros

def densify_curve(curve, samples_per_segment = 150):
    """
    Densify a curve by interpolating between points
    
    Args:
        curve: Original curve (N x d)
        samples_per_segment: Number of samples per segment
        
    Returns:
        Densified curve
    """
    dense = []

    if curve.shape[0] < curve.shape[1]:
        curve = curve.T
    
    for i in range(curve.shape[0] - 1):
        a = curve[i, :]
        b = curve[i + 1, :]
        t = xp.linspace(0, 1, samples_per_segment).reshape(-1, 1)
        segment = a + t * (b - a)
        dense.append(segment)
    
    return xp.vstack(dense).T

if __name__ == "__main__":
   pass