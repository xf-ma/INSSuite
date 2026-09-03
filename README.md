# INSSuite

Python implementations of four IMU-based inertial navigation models for distance measurement benchmarking:

- `HYBVIO`: full-state EKF
- `SOLA`: error-state Kalman filter following Sola's formulation
- `OPENSHOE`: OpenShoe-style error-state INS
- `MATERN`: proposed error-state Kalman filter with Matern 3/2 GP bias modeling

This repository is the codebase behind the paper **"Full-state versus error-state Kalman filters in IMU-based distance measurement"** to be presented in **October 2026** IPIN at Rome, Italy.

## What This Repo Does

The project compares different Kalman-filter formulations for short-range IMU-only distance measurement tasks.

The current workflow is:

1. Load IMU JSON files.
2. Calibrate accelerometer and gyroscope measurements.
3. Initialize attitude from an initial still segment.
4. Run one of the four navigation models.
5. Detect or inject ZUPT intervals.
6. Apply filtering and RTS smoothing.
7. Estimate trajectory and final traveled distance.
8. Evaluate error statistics across repeated trials.

## Models Included

| Model | Filter type | Original implementation / source | Frame convention | State size in code | Notes |
| --- | --- | --- | --- | --- |
| `HYBVIO` | Full-state EKF | C implementation; Solin et al. [1] | ENU | 19 | |
| `SOLA` | ESKF | Technical report; Sola [2] | ENU | 15 | |
| `OPENSHOE` | ESKF | MATLAB source; OpenShoe [3] | NED | 15 by default | Supports optional bias and scale-factor switches |
| `MATERN` | ESKF | Python implementation in this repository | ENU | 21 | Proposed model, uses Matern 3/2 GP for modelling IMU bias dynamics |

Original sources:

1. A. Solin, S. Cortes, E. Rahtu, and J. Kannala, "Inertial Odometry on Handheld Smartphones," *International Conference on Information Fusion*, pp. 1-5, 2018.
2. J. Sola, "Quaternion Kinematics for the Error-State KF," Technical Report, Institut de Robotica i Informatica Industrial (IRI), 2016.
3. [OpenShoe source repository](https://sourceforge.net/p/openshoe/omi/ci/master/tree/).

## Main Files

| File | Purpose |
| --- | --- |
| [`sc_main_distance_measure_batch.ipynb`](./sc_main_distance_measure_batch.ipynb) | Main experiment notebook for batch evaluation |
| [`run_navigation_trail.py`](./run_navigation_trail.py) | Entry point for running one trial |
| [`navigation_loop.py`](./navigation_loop.py) | Filtering, ZUPT update loop, and RTS smoothing |
| [`generic_filter_smoother.py`](./generic_filter_smoother.py) | Unified EKF/ESKF filter and smoother implementation |
| [`prepare_my_data.py`](./prepare_my_data.py) | Data loading, calibration, truth construction, plotting |
| [`sins.py`](./sins.py) | IMU container, strapdown INS utilities, ZUPT logic, plotting |
| [`ssm_hybvio.py`](./ssm_hybvio.py) | HybVIO's full-state EKF model |
| [`ssm_sola.py`](./ssm_sola.py) | Sola's error-state model |
| [`ssm_openshoe.py`](./ssm_openshoe.py) | OpenShoe's error-state model |
| [`ssm_matern.py`](./ssm_matern.py) | Proposed error-state model with Matern-bias |
| [`shape_distance.py`](./shape_distance.py) | Trajectory similarity metrics |
| [`xp.py`](./xp.py) | Unified `numpy`/`torch` backend wrapper (for later development) |

## Packages to Install

The codebase uses the following third-party Python packages.

### Required

- `numpy`
- `scipy`
- `plotly`
- `torch`
- `tqdm`
- `pandas`
- `jupyter`

### Optional

- `kaleido`
  Used only when exporting Plotly figures to images.

### Standard-library modules already used by the repo

No installation needed for:

- `json`
- `pathlib`
- `glob`
- `os`
- `copy`
- `time`
- `datetime`
- `threading`
- `types`
- `dataclasses`
- `typing`
- `abc`

### Install command

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy scipy plotly torch tqdm pandas jupyter
```

If you want static image export from Plotly:

```bash
pip install kaleido
```

## Data Format

The current pipeline expects IMU data in `.json` format. Based on [`sins.py`](./sins.py), each file should provide at least:

- `acc`: accelerometer samples, shape `(N, 3)` or `(3, N)`
- `gyr` or `gyros`: gyroscope samples, shape `(N, 3)` or `(3, N)`
- `mag`: magnetometer samples if available
- `ts`: sampling interval in seconds
- `timestamp` or `sT`
- `whichimu`
- `whereimu`
- `pathshape`

Optional fields already supported:

- `zupt_ref`: ZUPT flags provided by external information
- `sensor_name`
- `bias_acc`
- `bias_gyr`

The repository currently includes example data under:

- [`data/Xsens_A_100cm_3midstop`](./data/Xsens_A_100cm_3midstop)

This folder contains 20 JSON trials used by the batch notebook.

## Quick Start

### 1. Open the notebook

Use:

- [`sc_main_distance_measure_batch.ipynb`](./sc_main_distance_measure_batch.ipynb)

This is the main reproducible experiment entry point in the current repo.

### 2. Run the fixed-distance benchmark

The notebook is currently configured to evaluate:

- distance setting: `100cm_3midstop`
- models: `HYBVIO`, `SOLA`, `OPENSHOE`, `MATERN`
- input folder: `data/Xsens_A_100cm_3midstop`
- ZUPT detector: `STRICT`
- threshold: `0.005`
- minimum ZUPT duration: `1.5 s`
- initial still segment: `150` samples

For each trial, the notebook:

- loads one IMU JSON file,
- runs the chosen model,
- computes 2D and 3D endpoint distance,
- stores metrics to `results_<timestamp>/`,
- plots trajectories and error CDFs.

### 3. Switch models

Set `cfg.SSM_name` to one of:

- `"HYBVIO"`
- `"SOLA"`
- `"OPENSHOE"`
- `"MATERN"`

## Backend Notes

[`xp.py`](./xp.py) provides a unified wrapper over `numpy` and `torch`.

- Default behavior is `numpy`
- `torch` support later autodiff-related extensions
- the current notebook workflow can be understood as a standard Python scientific stack workflow


## Citation

If you use this repository, please cite the associated IPIN paper/project once the final bibliographic entry is available.

Provisional project title:

> Full-state versus error-state Kalman filters in IMU-based distance measurement
