import numpy as np
import xp
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass


@dataclass(slots=True)
class IMU:

    sensor_name: Optional[str] = None
    timestamp: Optional[np.ndarray] = None

    Ts: float = None
    acc: np.ndarray = None 
    gyr: np.ndarray = None 
    mag: np.ndarray = None 
    zupt_ref: np.ndarray = None 
    bias_acc: np.ndarray = None 
    bias_gyr: np.ndarray = None 

    
    whichimu: Optional[str] = None
    whereimu: Optional[str] = None

    pathshape: Optional[str] = None

    source_file: Optional[str] = None

    def _to_3byN(self, data):
        if data is None:
            return None
        data = np.asarray(data)

        if data.shape[1] == 3:
            data = data.T

        if data.shape[0] != 3:
            raise ValueError("Sensor data must have shape (3, N)")

        return data
    
    def from_dict(self, imu_dict):
        self.acc = self._to_3byN(imu_dict["acc"])
        self.gyr = self._to_3byN(imu_dict["gyros"]) if "gyros" in imu_dict else self._to_3byN(imu_dict["gyr"])
        self.mag = self._to_3byN(imu_dict["mag"])
        self.Ts = float(imu_dict["ts"])
        self.timestamp = np.asarray(imu_dict["sT"]) if "sT" in imu_dict else np.asarray(imu_dict["timestamp"])
        self.timestamp = self.timestamp - self.timestamp[0]
        self.zupt_ref = np.asarray(imu_dict["zupt_ref"]) if "zupt_ref" in imu_dict else None
        self.whichimu = imu_dict["whichimu"]
        self.whereimu = imu_dict["whereimu"]
        self.pathshape = imu_dict["pathshape"]
        self.sensor_name = imu_dict["sensor_name"] if "sensor_name" in imu_dict else None

        if self.whichimu=="simulated" or self.sensor_name=="simulated":
            self.bias_acc = xp.asarray(imu_dict["bias_acc"]).reshape(3,1)
            self.bias_gyr = xp.asarray(imu_dict["bias_gyr"]).reshape(3,1)

        # self.plot_imu(title="IMU data from input dict")

    def from_mat(self, imu_mat):
        self.acc = self._to_3byN(imu_mat["acc"])
        self.gyr = self._to_3byN(imu_mat["gyros"])
        self.mag = self._to_3byN(imu_mat["mag"])
        self.Ts = float(imu_mat["ts"])
        self.timestamp = np.asarray(imu_mat["sT"])
        self.timestamp = self.timestamp - self.timestamp
        self.whichimu = imu_mat["whichimu"]
        self.whereimu = imu_mat["whereimu"]
        self.pathshape = imu_mat["pathshape"]

        # self.plot_imu(title="IMU data from input .mat")
        # print(self)


    def trim(self, cut_start, cut_end):
        if cut_start is None:
            cut_start = 0
        if cut_end is None:
            cut_end = 0
        if cut_start < 0 or cut_end < 0 or cut_start > self.acc.shape[1] or cut_end > self.acc.shape[1]:
            raise ValueError("cut_start out of range")
        n = self.acc.shape[1]
        self.acc = self.acc[:, cut_start:n-cut_end] if self.acc is not None else None
        self.gyr = self.gyr[:, cut_start:n-cut_end] if self.gyr is not None else None
        self.mag = self.mag[:, cut_start:n-cut_end] if self.mag is not None else None
        self.zupt_ref = self.zupt_ref[cut_start:n-cut_end] if self.zupt_ref is not None else None
        self.timestamp = self.timestamp[cut_start:n-cut_end] if self.timestamp is not None else None

        self.plot_imu(title="After cutting")

    def detect_still_segments(self,
                            window_sec=1.0,
                            std_threshold=0.05,
                            mean_threshold=0.01):

        N = self.gyr.shape[1]
        window_size = int(window_sec * 1/self.Ts)

        norm = np.linalg.norm(self.gyr, axis=0)

        still_mask = np.zeros(N, dtype=bool)

        for i in range(0, N - window_size):
            window = norm[i:i + window_size]

            std_val = np.std(window)
            mean_val = np.mean(window)

            if std_val < std_threshold and mean_val < mean_threshold:
                still_mask[i:i + window_size] = True

        self.plot_imu(title='detected still intervals', zupt_in=still_mask)

        return still_mask
    
    def plot_imu(self, s=None, title=None, zupt_in=None, showfig=True):
        """
        Plot IMU acc, gyr, mag in 3 stacked subplots.
        Also plot magnetic field norm in the mag subplot.
        """
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except Exception as e:
            raise RuntimeError("plotly is required for plotting") from e

        if s is not None:
            acc = np.array(s.get("acc", None))
            gyr = np.array(s.get("gyr", None))
            mag = np.array(s.get("mag", np.zeros(acc.shape)))
            zupt = np.array(s.get("zupt_ref", None))
            if acc is None or gyr is None:
                raise ValueError("Expected at least 'acc', and 'gyr'in IMU dict.")
            if acc.shape[0] != 3:
                acc=acc.T
                gyr=gyr.T
                mag=mag.T
        else:
            acc = np.array(self.acc)
            gyr = np.array(self.gyr)
            mag = np.array(self.mag)
            zupt = np.array(self.zupt_ref)

        if zupt_in is not None:
            zupt = np.array(zupt_in)

        # timestamp = s.get("timestamp", None)
        # if isinstance(timestamp, list) and len(timestamp) >= n:
        #     t = timestamp[:n]
        #     x_label = "time"
        # else:
        x_label = "sample"

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("acc", "gyr", "mag"),
        )

        labels = ["x", "y", "z"]
        colors = ["red", "green", "blue"]
        for i in range(3):
            fig.add_trace(
                go.Scatter(y=acc[i,:], name=labels[i], line=dict(color=colors[i]),
                           showlegend=True,
                           legendgroup=labels[i]
                           ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(y=gyr[i,:], name=labels[i], line=dict(color=colors[i]),
                           showlegend=False,legendgroup=labels[i]
                           ),
                row=2,
                col=1,              
            )
            fig.add_trace(
                go.Scatter(y=mag[i,:], name=labels[i], line=dict(color=colors[i]),
                           showlegend=False,legendgroup=labels[i]
                           ),
                row=3,
                col=1,               
            )

        fig.add_trace(go.Scatter(y=np.linalg.norm(acc, axis=0), name="norm", 
                                 legendgroup="norm", showlegend=True, line=dict(width=2,color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(y=np.linalg.norm(gyr, axis=0), name="norm", 
                                 legendgroup="norm", showlegend=False, line=dict(width=2,color='black')), row=2, col=1)
        fig.add_trace(go.Scatter(y=np.linalg.norm(mag, axis=0), name="norm", 
                                 legendgroup="norm", showlegend=False, line=dict(width=2,color='black')), row=3, col=1)

        if  zupt is not None:
            fig.add_trace(go.Scatter(y=zupt, name="zupt", line=dict(width=2, color='black', dash='dash')), row=2, col=1)

        fig.update_layout(
            title=title or s.get("source_file", "IMU"),
            xaxis3_title=x_label,
            legend=dict(itemsizing="constant"),
            height=400,width=800
        )
        if showfig:
            fig.show()
        return fig


class SINS:
    """Strapdown Inertial Navigation System (SINS) class"""
    __slots__ = [
        "Ts",
        "sigma_a", "sigma_g", "Window_size",
        "pseudoVelocityR", "zuptR", "rotationZuptR", "initZuptR",
        "m", "quat", "delta_m", "P", "Q", "dfdx", "dfdq", "u", "m0", "P0", "Q0", "att0", "qua0",
        "ms", "quats", "delta_ms", "Ps",
        "measurement", "R", "dhdx", "dhdr",
        "g",
        "rotateorder", 
        "zupt_threshold", "ZVDtype", "zvd_k", "min_zupt_duration_s", 
        "truth", "truth_sparse",
        "estimator_type",
        "zupt_all", "zupt_T"
        "u_all",
        "metric_type",
        "nav_mode",
        "O3", "I3", "O43"
    ]
    
    def __init__(self, Ts):
        """
        Initialize SINS
        
        Args:
            Ts: Sampling period
        """
        self.Ts = Ts
        
        # Zero velocity detector settings
        self.sigma_a = 0.01  # Accelerometer noise std [m/s^2]
        self.sigma_g = 0.1 * xp.pi / 180  # Gyroscope noise std [rad/s]
        self.zupt_threshold = 1.5e5  # Threshold for SHOE zero-velocity detector
        self.Window_size = 3  # Window size [samples]
        
        # Measurement noise covariance settings
        self.pseudoVelocityR = 1e-4
        self.zuptR = 1e-6
        self.rotationZuptR = 1e-6
        self.initZuptR = 1e-4
        
        # State variables
        self.m = None
        self.quat = None
        self.delta_m = None
        self.P = None
        self.Q = None
        self.dfdx = None
        self.dfdq = None
        self.u = None
        self.m0 = None
        self.P0 = None
        self.Q0 = None
        self.att0 = None
        self.qua0 = None
        
        # Smoothed states
        self.ms = None
        self.quats = None
        self.delta_ms = None
        self.Ps = None
        
        # Measurement
        self.measurement = None
        self.R = None
        self.dhdx = None
        self.dhdr = None
        
        # Gravity initialization
        altitude = xp.array(100)  # meters
        latitude = xp.array(58 * xp.pi / 180)  # degrees to radians
        gamma = xp.array(9.780327 * (1 + 0.0053024 * xp.sin(latitude)**2 - 
                           0.0000058 * xp.sin(2*latitude)**2))
        self.g = gamma - ((3.0877e-6) - (0.004e-6) * xp.sin(latitude)**2) * altitude + \
            (0.072e-12) * altitude**2
        
        # Settings
        self.rotateorder = 'zyx'
        self.ZVDtype = "SHOE"
        self.zvd_k = None
        self.min_zupt_duration_s = 0.0
        self.truth = None
        self.truth_sparse = None
        self.estimator_type = "EKF"
        self.zupt_all = None
        self.zupt_T = None
        self.u_all = None
        self.metric_type = "END"
        self.nav_mode = True  # True for navigation, False for parameter estimation

        # other constants 
        self.O3 = xp.zeros((3, 3))
        self.I3 = xp.eye(3)
        self.O43 = xp.zeros((4, 3))

    def tag(self, name, x):
        if type(x) == 'tensor':
            print(f"[TAG] {name}: requires_grad={x.requires_grad}, grad_fn={x.grad_fn}")
        else:
            pass
        return x
    
    def re_initialize(self):
        """Reinitialize state to initial values"""
        self.m = xp.copy(self.m0)
        self.P = xp.copy(self.P0)
        self.Q = xp.copy(self.Q0)
        self.R = xp.copy(self.R0)
        if hasattr(self, 'quat'):
            self.quat = xp.copy(self.qua0)
    
    
    def quat_from_two_vectors(self, a, b):
        """
        Compute quaternion from two vectors
        
        Args:
            a, b: 3D vectors
            
        Returns:
            Dictionary with 'w' and 'vec' keys
        """
        axis = xp.cross(a, b)
        axis = axis / xp.norm(axis)
        theta = xp.arccos(xp.dot(a, b) / (xp.norm(a) * xp.norm(b)))
        
        return {
            'w': xp.cos(theta / 2),
            'vec': axis * xp.sin(theta / 2)
        }
    
    def dcm(self, axis, angle):
        if axis.lower()=='x':
            Cbn = np.array([[1,0,0],[0,np.cos(angle),np.sin(angle)],[0,-np.sin(angle),np.cos(angle)]])
        elif axis.lower()=='y':
            Cbn = np.array([[np.cos(angle),0, -np.sin(angle)],[0, 1, 0],[np.sin(angle), 0, np.cos(angle)]])
        elif axis.lower()=='z':
            Cbn = np.array([[np.cos(angle), np.sin(angle), 0],[-np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
        else:
            raise ValueError('axis wrong')
        return Cbn

    def quat2rmat(self, q):
        """
        Convert quaternion to rotation matrix
        
        Args:
            q: Quaternion [qw, qx, qy, qz]
            
        Returns:
            3x3 rotation matrix
        """
        q = q.flatten()
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        R = xp.array([
            [qw*qw + qx*qx - qy*qy - qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
            [2*qx*qy + 2*qw*qz, qw*qw - qx*qx + qy*qy - qz*qz, 2*qy*qz - 2*qw*qx],
            [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, qw*qw - qx*qx - qy*qy + qz*qz]
        ])
        
        return R
    
    def quat2rmat_d(self, q):
        """
        Compute rotation matrix and its derivatives w.r.t. quaternion
        
        Args:
            q: Quaternion [qw, qx, qy, qz]
            
        Returns:
            R: Rotation matrix
            dR: List of 4 derivative matrices
        """
        q = q.flatten()
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        dR = [None] * 4
        
        dR[0] = xp.array([
            [2*qw, -2*qz, 2*qy],
            [2*qz, 2*qw, -2*qx],
            [-2*qy, 2*qx, 2*qw]
        ])
        
        dR[1] = xp.array([
            [2*qx, 2*qy, 2*qz],
            [2*qy, -2*qx, -2*qw],
            [2*qz, 2*qw, -2*qx]
        ])
        
        dR[2] = xp.array([
            [-2*qy, 2*qx, 2*qw],
            [2*qx, 2*qy, 2*qz],
            [-2*qw, 2*qz, -2*qy]
        ])
        
        dR[3] = xp.array([
            [-2*qz, -2*qw, 2*qx],
            [2*qw, -2*qz, 2*qy],
            [2*qx, 2*qy, 2*qz]
        ])
        
        R = self.quat2rmat(q)
        
        return R, dR
    
    def quaternion_product(self, q1, q2):
        """
        Compute quaternion product (Hamilton product)
        
        Args:
            q1, q2: Quaternions [qw, qx, qy, qz]
            
        Returns:
            Product quaternion
        """
        q1 = q1.flatten()
        q1w, q1x, q1y, q1z = q1[0], q1[1], q1[2], q1[3]
        
        q1L = xp.array([
            [q1w, -q1x, -q1y, -q1z],
            [q1x, q1w, -q1z, q1y],
            [q1y, q1z, q1w, -q1x],
            [q1z, -q1y, q1x, q1w]
        ])
        
        return q1L @ q2
    
    def rotation_vec2quaternion(self, v):
        """
        Convert rotation vector to quaternion
        
        Args:
            v: 3D rotation vector
            
        Returns:
            Quaternion [qw, qx, qy, qz]
        """
        angle = xp.norm(v)
        
        if angle < 1e-10:
            return xp.array([1, 0, 0, 0])
        
        axis = v / angle
        qw = xp.cos(angle / 2)
        qv = axis * xp.sin(angle / 2)
        
        return xp.vstack([xp.array([qw]), qv])
    
    def rotation_vec2rotation_matrix(self, v):
        """
        Convert rotation vector to rotation matrix (Rodrigues formula)
        
        Args:
            v: 3D rotation vector
            
        Returns:
            3x3 rotation matrix
        """
        angle = xp.norm(v)
        axis = v / angle
        
        R = (xp.eye(3) * xp.cos(angle) + 
             xp.sin(angle) * self.skew(axis) + 
             xp.outer(axis, axis) * (1 - xp.cos(angle)))
        
        return R
    
    def skew(self, v):
        """
        Create skew-symmetric matrix from vector (right-handed)
        
        Args:
            v: 3D vector
            
        Returns:
            3x3 skew-symmetric matrix
        """
        v = v.flatten()
        return xp.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def normalize_quaternions(self):
        """Normalize quaternions to unit length"""      
        if hasattr(self, 'QUA') and self.QUA is not None:
            m_old = xp.copy(self.m)
            q_old = xp.copy(self.m[self.QUA, :])
            m_old[self.QUA, :] = q_old / xp.norm(q_old, axis=0)
            self.m = m_old
        elif self.quat is not None:
            q_old = xp.copy(self.quat)
            self.quat = q_old / xp.norm(q_old)
        else:
            raise ValueError("Where is the quaternion?")
    
    def lock_biases(self):
        """Lock bias states in covariance matrix"""
        bias_indices = xp.concatenate([self.BGA, self.BAA, self.BAT])
        self.P[bias_indices, :] = 0
        self.P[:, bias_indices] = 0
    
    def results_container(self, data_length):
        """
        Create container for storing results
        
        Args:
            data_length: Number of time steps
            
        Returns:
            Dictionary with allocated arrays
        """
        results = {}
        
        for i, name in enumerate(self.STATE_PART_NAMES):
            results[name] = xp.zeros((self.STATE_PART_SIZES[i], data_length))
        
        results['ATT'] = xp.zeros((3, data_length))
        results['QUA'] = xp.zeros((4, data_length))
        results['COV'] = xp.zeros((self.stateDim, self.stateDim, data_length))
        results['Fx'] = xp.zeros((self.stateDim, self.stateDim, data_length))
        results['m'] = xp.zeros((self.stateDim, data_length))
        results['delta_m'] = xp.zeros((self.stateDim, data_length))
        results['delta_ms'] = xp.zeros((self.stateDim, data_length))
        results['ms'] = xp.zeros((self.stateDim, data_length))
        results['Ps'] = xp.zeros((self.stateDim, self.stateDim, data_length))
        
        return results
    
    def store_result(self, container, t):
        """
        Store current state in results container
        
        Args:
            container: Results dictionary
            t: Time index
        """
        container['POS'][:, t] = self.position().ravel() 
        container['VEL'][:, t] = self.velocity().ravel() 
        
        if hasattr(self, 'QUA'):
            container['QUA'][:, t] = self.orientation().ravel() 
            container['ATT'][:, t] = self.qua2att(self.orientation()).ravel() 
        else:
            container['ATT'][:, t] = self.orientation().ravel() 
            container['QUA'][:, t] = self.quat.ravel() 
        
        container['BGA'][:, t] = self.bias_gyroscope_additive().ravel() 
        container['BAA'][:, t] = self.bias_accelerometer_additive().ravel() 
        container['COV'][:, :, t] = self.get_state_covariance()
        container['m'][:, t] = self.m.ravel() 
        container['delta_m'][:, t] = self.delta_m.ravel() 
        container['Fx'][:, :, t] = self.dfdx
    
    def zero_velocity_detector(self, u):
        """
        Detect zero velocity periods
        
        Args:
            u: IMU data (6 x N array)
            
        Returns:
            zupt: Binary array (1 = zero velocity, 0 = moving)
            T: Test statistics
        """
        if 'SHOE' in self.ZVDtype.upper():
            zupt, T = self._shoe_detector(u)
        elif self.ZVDtype.upper() == "STRICT":
            zupt, T = self._strict_detector(u)
        else:
            raise ValueError("Undefined zero velocity detector type!")
        return self._post_process_zupt(zupt), T#-T[1]
        
    def plot_zupt_detection(self, zupt_in=None, u={'':None}, showfig=True, title=None):
        """
        Plot ZUPT detection results
        """

        zupt_all=np.asarray(self.zupt_all)

        fig = go.Figure()

        for lbl, data in u.items():
            if data is not None:
                data_np = xp.as_numpy(data)
                if len(data_np.shape) == 2:
                    dim = min(data_np.shape)
                    data_np = data_np.reshape(-1,dim)
                    for i in range(dim):
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(max(data_np.shape))),
                                y=data_np[:,i],
                                mode='lines',
                                name=lbl+str(i),
                                line=dict(width=2)
                            )
                )
                elif len(data_np.shape) == 1:
                    fig.add_trace(
                            go.Scatter(
                                x=list(range(len(data))),
                                y=data_np,
                                mode='lines',
                                name=lbl,
                                line=dict(width=2)
                            ))
                else:
                    pass
                

        if zupt_all is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(zupt_all))),
                    y=xp.as_numpy(zupt_all)*xp.as_numpy(self.zupt_threshold),
                    mode='lines',
                    name='ZUPT flags',
                    line=dict(width=2)
                )
            )
        else:
            zupt_all = np.array([1,1,1,1,1])
        if zupt_in is not None:
            zupt_in = np.asarray(zupt_in)
            if zupt_in.shape == self.zupt_all.shape:
                if not np.array_equal(zupt_in, self.zupt_all):
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(zupt_all))),
                            y=xp.as_numpy(zupt_in)*xp.as_numpy(self.zupt_threshold),
                            name="ZUPT (external)",
                            line=dict(width=1),
                            mode='lines'
                        ),
                    )
                    print("External ZUPT detection added.")
            else:
                print(f"\033[33m[WARNING] External ZUPT data shape {zupt_in.shape} does not match internal data shape {self.zupt_all.shape}. Skipping plotting.\033[0m")

        if self.zupt_T is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.zupt_T))),
                    y=xp.as_numpy(self.zupt_T),
                    mode='lines+markers', 
                    name='ZUPT test statistic',
                    line=dict(width=1.5),
                    marker=dict(
                        size=3,          
                        symbol='circle', 
                        line=dict(width=1, color='white')
                    )
                )
            )

        if self.zupt_threshold is not None:
            gamma_val = xp.as_numpy(self.zupt_threshold)
            fig.add_trace(
                go.Scatter(
                    x=[0, len(zupt_all)-1],
                    y=[gamma_val, gamma_val],
                    mode='lines',
                    line=dict(color='black', width=2, dash='dot'),
                    name=f'Threshold ({gamma_val:.3f})',
                    showlegend=True
                )
            )
            

        fig.update_layout(
            title=dict(
                text=title or 'ZUPT Detection',
                font=dict(size=16)
            ),
            xaxis_title='Sample',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            width=800,
            height=300
        )
        if showfig:
            fig.show()

    def _find_static_intervals(self, static_flags):
        """
        Find start/end indices for contiguous static intervals from binary flags.
        """
        static_flags = np.asarray(static_flags).astype(bool).reshape(-1)
        static_padded = np.concatenate(([False], static_flags, [False]))
        changes = np.diff(static_padded.astype(int))
        start_indices = np.where(changes == 1)[0]
        end_indices = np.where(changes == -1)[0] - 1
        return start_indices, end_indices

    def _compute_interval_means_from_indices(self, state, start_indices, end_indices):
        """
        Compute interval means from state and static interval bounds.
        Supports state shape (M, N) or (N, M). Returns means as (M, K).
        """
        state = np.asarray(state)
        start_indices = np.asarray(start_indices, dtype=int).reshape(-1)
        end_indices = np.asarray(end_indices, dtype=int).reshape(-1)
        if state.ndim != 2:
            raise ValueError("state must be a 2D array")
        if start_indices.size != end_indices.size:
            raise ValueError("start_indices and end_indices must have same length")

        num_intervals = start_indices.size
        if num_intervals == 0:
            return np.zeros((state.shape[0], 0))

        max_end = int(np.max(end_indices))
        if state.shape[1] > max_end:  # state is (M, N)
            state_dim = state.shape[0]
            interval_means = np.zeros((state_dim, num_intervals))
            for i in range(num_intervals):
                interval_means[:, i] = np.mean(
                    state[:, start_indices[i]:end_indices[i] + 1], axis=1
                )
        elif state.shape[0] > max_end:  # state is (N, M)
            state_dim = state.shape[1]
            interval_means = np.zeros((state_dim, num_intervals))
            for i in range(num_intervals):
                interval_means[:, i] = np.mean(
                    state[start_indices[i]:end_indices[i] + 1, :], axis=0
                )
        else:
            raise ValueError("state length does not match interval indices")

        return interval_means

    def _compute_static_interval_means(self, state, static_flags):
        """
        Compatibility helper: find static intervals and compute interval means.
        """
        start_indices, end_indices = self._find_static_intervals(static_flags)
        interval_means = self._compute_interval_means_from_indices(
            state,
            start_indices,
            end_indices
        )
        return interval_means, start_indices, end_indices

    def _to_3xN(self, points, name="points"):
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        if arr.shape[0] in (2, 3):
            out = arr
        elif arr.shape[1] in (2, 3):
            out = arr.T
        else:
            raise ValueError(f"{name} must have 2 or 3 coordinates per sample")
        if out.shape[0] == 2:
            out = np.vstack([out, np.zeros((1, out.shape[1]))])
        return out

    def _fit_rigid_transform(self, src_points, dst_points):
        """
        Fit rigid transform dst ~= R @ src + t using SVD (Kabsch).
        """
        src = self._to_3xN(src_points, name="src_points")
        dst = self._to_3xN(dst_points, name="dst_points")
        if src.shape[1] != dst.shape[1]:
            raise ValueError("src_points and dst_points must have the same number of samples")
        n = src.shape[1]
        if n == 0:
            return np.eye(3), np.zeros((3, 1))
        if n == 1:
            return np.eye(3), dst[:, [0]] - src[:, [0]]

        src_centroid = np.mean(src, axis=1, keepdims=True)
        dst_centroid = np.mean(dst, axis=1, keepdims=True)
        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid

        H = src_centered @ dst_centered.T
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T
        t = dst_centroid - R @ src_centroid
        return R, t

    def _align_estimated_with_truth_static_points(
        self,
        estimated_pos,
        static_flags,
        points_number_for_align=4
    ):
        """
        Align estimated trajectory to truth static points using the first X static points.
        """
        est = self._to_3xN(estimated_pos, name="estimated_pos")
        flags = np.asarray(static_flags).astype(bool).reshape(-1)
        if flags.size != est.shape[1]:
            return est, np.zeros((3, 0)), None, 0, np.eye(3), np.zeros((3, 1))
        interval_means, _, _ = self._compute_static_interval_means(est, flags)

        if self.truth_sparse is None:
            return est, interval_means, None, 0, np.eye(3), np.zeros((3, 1))

        x_use = max(int(points_number_for_align), 0)
        n_use = min(x_use, interval_means.shape[1], self.truth_sparse.shape[1])
        if n_use <= 0:
            return est, interval_means, self.truth_sparse, 0, np.eye(3), np.zeros((3, 1))

        R, t = self._fit_rigid_transform(
            interval_means[:, :n_use],
            self.truth_sparse[:, :n_use]
        )
        est_aligned = R @ est + t
        interval_means_aligned = R @ interval_means + t
        return est_aligned, interval_means_aligned, self.truth_sparse, n_use, R, t

    
    def plot_trajectory_zupt_only(
        self,
        results,
        title=None,
        points_number_for_align=4,
        truth_group_distance=None,
        showfig=True
    ):
        if self.zupt_all is None:
            raise ValueError("self.zupt_all is None; run ZUPT detection before plotting ZUPT-only trajectory")
        if self.truth is None:
            raise ValueError("self.truth is None; truth is required for aligned ZUPT-only plotting")

        idx = np.asarray(self.zupt_all).astype(bool)
        pos = self._to_3xN(results['POS'], name="results['POS']")
        if idx.size != pos.shape[1]:
            raise ValueError(f"ZUPT length {idx.size} does not match trajectory length {pos.shape[1]}")
        interval_means, _, _ = self._compute_static_interval_means(pos, idx)
        truth_grouped = None
        n_used = 0
        if self.truth_sparse is not None and idx.size > 0 and points_number_for_align>0:
            pos, interval_means, truth_grouped, n_used, _, _ = self._align_estimated_with_truth_static_points(
                estimated_pos=pos,
                static_flags=idx,
                points_number_for_align=points_number_for_align
            )

        # trick
        # interval_means = np.hstack((pos[:,0].reshape(3,1),pos[:,-1].reshape(3,1)))

        colors = {'truth':"#30a14a",'zupt': '#ff7f0e'}

        # ======================
        # 3D Plot
        # ======================

        # True trajectory
        truth_np = self._to_3xN(self.truth, name="truth")
        truth_sparse_np = None if self.truth_sparse is None else self._to_3xN(self.truth_sparse, name="truth_sparse")
        z_truth = truth_np[2, :]
        
        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=truth_np[0, :],
                y=truth_np[1, :],
                z=z_truth,
                mode="lines",
                name="True Trajectory",
                line=dict(color=colors['truth'])
            )
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=[truth_np[0, 0]],
                y=[truth_np[1, 0]],
                z=[z_truth[0]],
                mode="markers",
                name="Start Point",
                marker=dict(size=4, color="green")
            )
        )

        if truth_grouped is not None and truth_grouped.shape[1] > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=truth_grouped[0, :],
                    y=truth_grouped[1, :],
                    z=truth_grouped[2, :],
                    mode="markers+lines",
                    name="True static points (grouped)",
                    marker=dict(size=6,color=colors['truth'])
                )
            )
        else:
            if truth_sparse_np is not None and truth_sparse_np.shape[1] > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=truth_sparse_np[0, :],
                        y=truth_sparse_np[1, :],
                        z=truth_sparse_np[2, :],
                        mode="markers",
                        name="True static points",
                        marker=dict(size=6,color=colors['truth'])
                    )
                )

        # Interval means
        if interval_means.shape[1] > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=interval_means[0, :],
                    y=interval_means[1, :],
                    z=interval_means[2, :],
                    mode="markers+lines",
                    name="ZUPT Interval Means",
                    marker=dict(size=3,color=colors['zupt'])
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="x [m]",
                yaxis_title="y [m]",
                zaxis_title="z [m]",
                # aspectmode="data"
            ),
            title=title or (
                f"Positions at ZUPT (aligned with first {n_used} static points)"
                if n_used > 0 else
                "Positions at ZUPT"
            ),
            showlegend=True,
            height=600,
            width=600
        )
        if showfig:
            fig.show()

        return interval_means, fig

    def plot_zupt_points_error(
        self,
        pos_zupt_est,
        pos_zupt_truth=None,
        title=None,
        showfig=True
    ):
        
        if pos_zupt_truth is None and self.truth_sparse is not None:
            pos_zupt_truth = self._to_3xN(self.truth_sparse, name="truth_sparse")  

        # Convert inputs to numpy arrays
        est = np.asarray(pos_zupt_est, dtype=float)
        truth = np.asarray(pos_zupt_truth, dtype=float)

        if truth.shape[1]!=est.shape[1]:
            print(f"Not support plot_zupt_points_error, number of zupt points is {est.shape[1]} but turth is {truth.shape[1]}")
            return

        # Compute 2D (xy-plane) distance error for each ZUPT point
        err_2d = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)
        # Compute 3D distance error for each ZUPT point
        err_3d = np.linalg.norm(est[:, :3] - truth[:, :3], axis=1)

        # Mean errors over all ZUPT points
        mean_2d = np.mean(err_2d)
        mean_3d = np.mean(err_3d)

        # X axis: ZUPT point index
        idx = np.arange(len(err_2d))

        # Create two subplots: one for 2D error, one for 3D error
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                f"2D (xy-plane) Distance Error | Mean = {mean_2d:.4f}",
                f"3D Distance Error | Mean = {mean_3d:.4f}"
            )
        )

        # 2D error line plot
        fig.add_trace(
            go.Scatter(
                x=idx, y=err_2d,
                mode="lines+markers",
                name="2D Error"
            ),
            row=1, col=1
        )

        # 3D error line plot
        fig.add_trace(
            go.Scatter(
                x=idx, y=err_3d,
                mode="lines+markers",
                name="3D Error"
            ),
            row=2, col=1
        )

        # Axis labels
        fig.update_xaxes(title_text="ZUPT Point Index", row=1, col=1)
        fig.update_xaxes(title_text="ZUPT Point Index", row=2, col=1)
        fig.update_yaxes(title_text="Distance Error", row=1, col=1)
        fig.update_yaxes(title_text="Distance Error", row=2, col=1)

        # Overall figure title
        fig.update_layout(
            title_text=title if title is not None else "ZUPT Points Distance Error"
        )

        if showfig:
            fig.show()

        return fig
        


    def _shoe_detector(self, u):
        """SHOE zero velocity detector"""
        print("Using SHOE detector")

        N = u.shape[1]
        zupt = xp.zeros(N)
        
        g = xp.norm(self.gravity)
        sigma2_a = self.sigma_a**2
        sigma2_g = self.sigma_g**2
        W = self.Window_size
        
        T = xp.zeros(N - W + 1)
        
        for k in range(N - W + 1):
            ya_m = xp.mean(u[0:3, k:k+W], axis=1)
            
            for l in range(k, k + W):
                tmp = u[0:3, l] - g * ya_m / xp.norm(ya_m)
                T[k] += (u[3:6, l] @ u[3:6, l] / sigma2_g + 
                        tmp @ tmp / sigma2_a)
        
        T = T / W

        for k in range(len(T)):
            if T[k] < self.zupt_threshold:
                zupt[k:k+W] = 1
        pad = int(xp.floor(W/2))
        T = xp.concatenate((xp.ones(pad)*xp.max(T), T, xp.ones(pad)*xp.max(T)))

        return zupt, T
    
    def _strict_detector(self, u):
        """Strict zero velocity detector"""
        print("Using strict detector ZUPT")

        N = u.shape[1]
        W = self.Window_size
        zupt = xp.zeros(N, dtype=int)
        T = xp.zeros(N - W + 1)
        
        for k in range(N - W + 1):
            T[k] = float(xp.as_numpy(u[5, k:k+W]).std())
            
            if T[k] < self.zupt_threshold:
                zupt[k] = 1

        return zupt, T

    def _post_process_zupt(self, zupt):
        zupt_out = xp.copy(zupt)
        if self.min_zupt_duration_s and self.min_zupt_duration_s > 0:
            min_samples = int(round(float(self.min_zupt_duration_s) / float(self.Ts)))
            min_samples = max(min_samples, 1)
            zupt_out = self.remove_short_stance_intervals(zupt_out, min_samples)
        return zupt_out

    def remove_short_stance_intervals(self, zupt, min_samples):
        """
        Remove stance runs shorter than min_samples by converting them to moving.
        """
        zupt_np = xp.as_numpy(zupt).astype(int).copy()
        n = len(zupt_np)
        i = 0
        while i < n:
            if zupt_np[i] == 1:
                j = i
                while j < n and zupt_np[j] == 1:
                    j += 1
                if (j - i) < min_samples:
                    zupt_np[i:j] = 0
                i = j
            else:
                i += 1
        return xp.asarray(zupt_np)
    
    def qua2att(self, q):
        """
        Convert quaternion to Euler angles
        
        Args:
            q: Quaternion [qw, qx, qy, qz]
            
        Returns:
            att: Euler angles [pitch, roll, yaw] in radians
        """
        Cbn = self.qua2dcm(q, 'Cbn')
        order = self.rotateorder.lower()
        
        if order == 'zxy':
            att = xp.array([
                xp.arcsin(Cbn[1, 2]),
                xp.arctan2(-Cbn[0, 2], Cbn[2, 2]),
                xp.arctan2(-Cbn[1, 0], Cbn[1, 1])
            ])
        elif order == 'zyx':
            att = xp.array([
                xp.arctan2(Cbn[1, 2], Cbn[2, 2]),
                xp.arcsin(-Cbn[0, 2]),
                xp.arctan2(Cbn[0, 1], Cbn[0, 0])
            ])
        else:
            raise ValueError('Undefined rotation order')
        
        return xp.reshape(att, (3, 1))
    
    def qua2dcm(self, q, Ctype):
        """
        Convert quaternion to direction cosine matrix
        
        Args:
            q: Quaternion [qw, qx, qy, qz]
            Ctype: 'Cnb' or 'Cbn'
            
        Returns:
            3x3 DCM
        """
        q = q.flatten()
        q11 = q[0] * q[0]
        q12 = q[0] * q[1]
        q13 = q[0] * q[2]
        q14 = q[0] * q[3]
        q22 = q[1] * q[1]
        q23 = q[1] * q[2]
        q24 = q[1] * q[3]
        q33 = q[2] * q[2]
        q34 = q[2] * q[3]
        q44 = q[3] * q[3]
        
        if Ctype == 'Cnb':
            C = xp.array([
                [q11 + q22 - q33 - q44, 2*(q23 - q14), 2*(q24 + q13)],
                [2*(q23 + q14), q11 - q22 + q33 - q44, 2*(q34 - q12)],
                [2*(q24 - q13), 2*(q34 + q12), q11 - q22 - q33 + q44]
            ])
        elif Ctype == 'Cbn':
            C = xp.array([
                [q11 + q22 - q33 - q44, 2*(q23 + q14), 2*(q24 - q13)],
                [2*(q23 - q14), q11 - q22 + q33 - q44, 2*(q34 + q12)],
                [2*(q24 + q13), 2*(q34 - q12), q11 - q22 - q33 + q44]
            ])
        else:
            raise ValueError('Invalid DCM type')
        
        return C
    
    def init_attitude_from_acc(self, Acc, yaw0=0):
        """
        Initialize attitude from accelerometer measurements
        
        Args:
            Acc: Accelerometer data (3 x N array)
            yaw0: Initial yaw angle (default: 0)
            
        Returns:
            qua: Quaternion
            att: Euler angles [pitch, roll, yaw]
        """
        fx = xp.mean(Acc[0, :])
        fy = xp.mean(Acc[1, :])
        fz = xp.mean(Acc[2, :])
        
        order = self.rotateorder
        
        if order == 'zyx':
            pitch = -xp.arctan2(fy, xp.sqrt(fx**2 + fz**2))
            roll  = xp.arctan2(fx, fz)
        elif order == 'zxy':
            pitch = xp.arcsin(fy / xp.sqrt(fx**2 + fy**2 + fz**2))
            roll = xp.arctan2(-fx, fz)
        else:
            raise ValueError('Undefined rotation order')
        
        if yaw0 != 0:
            att = xp.array([[pitch], [roll,] [yaw0]])
        else:
            att = xp.array([[pitch], [roll], [0]])
        
        qua = self.att2qua(att)
        
        # print(f'----- Initial attitude (deg): {att.flatten()*180/xp.pi} -----')
        
        return xp.reshape(qua,(4,1)), xp.reshape(att,(3,1))
    
    def att2qua(self, att):
        """
        Convert Euler angles to quaternion
        
        Args:
            att: Euler angles [pitch, roll, yaw]
            
        Returns:
            Quaternion [qw, qx, qy, qz]
        """
        att2 = att / 2
        s = xp.sin(att2)
        c = xp.cos(att2)
        sx, sy, sz = s[0], s[1], s[2]
        cx, cy, cz = c[0], c[1], c[2]
        
        order = self.rotateorder
        
        if order == 'zxy':
            qsn = xp.array([
                [cx*cy*cz - sx*sy*sz],
                [sx*cy*cz - cx*sy*sz],
                [cx*sy*cz + sx*cy*sz],
                [sx*sy*cz + cx*cy*sz]
            ])
        elif order == 'zyx':
            qsn = xp.array([
                [cx*cy*cz + sx*sy*sz],
                [sx*cy*cz - cx*sy*sz],
                [cx*sy*cz + sx*cy*sz],
                [cx*cy*sz - sx*sy*cz]
            ])
        else:
            raise ValueError('Undefined rotation order')
        
        return qsn
    
    def dcm(self, axis, angle):
        """
        Create direction cosine matrix for rotation about axis
        
        Args:
            axis: 'x', 'y', or 'z'
            angle: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        if axis == 'x':
            return xp.array([
                [1, 0, 0],
                [0, xp.cos(angle), xp.sin(angle)],
                [0, -xp.sin(angle), xp.cos(angle)]
            ])
        elif axis == 'y':
            return xp.array([
                [xp.cos(angle), 0, -xp.sin(angle)],
                [0, 1, 0],
                [xp.sin(angle), 0, xp.cos(angle)]
            ])
        elif axis == 'z':
            return xp.array([
                [xp.cos(angle), xp.sin(angle), 0],
                [-xp.sin(angle), xp.cos(angle), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError('Invalid axis')
    
    def error_state_reset(self, dx, P):
        """
        Reset error state covariance after update
        
        Args:
            dx: Error state
            P: Covariance matrix
            
        Returns:
            Updated covariance matrix
        """
        G = xp.eye(self.stateDim)
        G[self.ATT, self.ATT] = xp.eye(3) - 0.5 * self.skew(dx[self.ATT, :])
        return G @ P @ G.T
    
    # Plotting methods
    def plot_results(self, results):
        """
        Plot navigation results
        
        Args:
            results: Results dictionary
        """
        fig_p = self.plot_trajectory(results)
        fig_q = self.plot_quaternion(results)
        fig_att = self.plot_attitude(results)

        return fig_p, fig_q, fig_att

    
    def plot_trajectory(
        self,
        results,
        axis2D=(0,1),
        title=None,
        points_number_for_align=4,
        truth_group_distance=None,
        showfig=True
    ):
        # """Plot 2D and 3D trajectories"""
        pos_np = self._to_3xN(xp.as_numpy(results['POS']), name="results['POS']")
        pos_smth_np = self._to_3xN(xp.as_numpy(results['ms'][0:3, :]), name="results['ms'][0:3, :]")
        truth_grouped = None
        n_used = 0
        if self.truth_sparse is not None and self.zupt_all is not None and points_number_for_align>0:
            idx = np.asarray(self.zupt_all).astype(bool)
            if idx.size == pos_np.shape[1]:
                pos_np, _, truth_grouped, n_used, R_align, t_align = self._align_estimated_with_truth_static_points(
                    estimated_pos=pos_np,
                    static_flags=idx,
                    points_number_for_align=points_number_for_align
                )
                pos_smth_np = R_align @ pos_smth_np + t_align

        fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('2D View', '3D View'),
                        specs=[[{'type': 'xy'}, {'type': 'scene'}]],
                        column_widths=[0.4, 0.6]
                    )

        colors = {
            'truth': "#3e923e",
            'filtered': "#6a8cfd",
            'smoothed': "#e2ad4b",
            'start': "#B6B6B6",
            'end': "#424242"
        }

        traces_2d = []
        if self.truth is not None:
            truth_np = self._to_3xN(xp.as_numpy(self.truth), name="truth")
            traces_2d.append(
                go.Scatter(
                    x=truth_np[axis2D[0], :], y=truth_np[axis2D[1], :],
                    mode='lines',
                    line=dict(color=colors['truth'], width=3), 
                    name='True trajectory', legendgroup='True trajectory', showlegend=False,
                )
            )
            traces_2d.append(
                go.Scatter(x=[truth_np[axis2D[0], 0]], y=[truth_np[axis2D[1],0]],
                        mode='markers', name='Start', legendgroup='Start', showlegend=False,
                        marker=dict(size=5, color=colors['start'], symbol='diamond')),
            )
        if self.truth_sparse is not None:
            truth_sparse_np = truth_grouped if truth_grouped is not None else self._to_3xN(xp.as_numpy(self.truth_sparse), name="truth_sparse")
            traces_2d.append(
                go.Scatter(
                    x=truth_sparse_np[axis2D[0], :], y=truth_sparse_np[axis2D[1], :],
                    mode='markers',
                    marker=dict(color=colors['truth'], size=5),
                    name='True static points', legendgroup='True static points', showlegend=False
                )
            )

        traces_2d.append(
            go.Scatter(
                x=pos_np[0, :], y=pos_np[1, :],
                mode='lines+markers',
                line=dict(color=colors['filtered'], width=1),
                marker=dict(size=2), 
                name='Estimation', legendgroup='Estimation', showlegend=False,
            )
        )

        traces_2d.append(
            go.Scatter(
                x=pos_smth_np[0, :], y=pos_smth_np[1, :],
                mode='lines+markers',
                line=dict(color=colors['smoothed'], width=1),
                marker=dict(size=2), 
                name='Smoothed estimation', legendgroup='Smoothed estimation', showlegend=False,
            )
        )


        traces_3d = []
        if self.truth is not None:
            truth_np = self._to_3xN(xp.as_numpy(self.truth), name="truth")
            z_truth = truth_np[2, :]
            traces_3d.append(
                go.Scatter3d(
                    x=truth_np[0, :], y=truth_np[1, :], z=z_truth,
                    mode='lines', name='True trajectory', legendgroup='True trajectory', showlegend=True,
                    line=dict(color=colors['truth'], width=3)
                )
            )
            traces_3d.append(
                go.Scatter3d(
                    x=[truth_np[0, 0]], y=[truth_np[1, 0]], z=z_truth,
                    mode='markers', name='Start', legendgroup='Start', showlegend=False,
                    marker=dict(size=5, color=colors['start'], symbol='diamond')
                )
            )

        if self.truth_sparse is not None:
            truth_sparse_np = truth_grouped if truth_grouped is not None else self._to_3xN(xp.as_numpy(self.truth_sparse), name="truth_sparse")
            if truth_sparse_np.shape[0] == 2:
                z_truth_sparse = np.zeros(truth_sparse_np.shape[1])
            elif truth_sparse_np.shape[0] == 3:
                z_truth_sparse = truth_sparse_np[2, :]
            else:
                raise ValueError("Check the size of truth_sparse")
            traces_3d.append(
                go.Scatter3d(
                    x=truth_sparse_np[0, :], y=truth_sparse_np[1, :], z=z_truth_sparse,
                    mode='markers', name='True static points', legendgroup='True static points', showlegend=True,
                    marker=dict(color=colors['truth'], size=5)
                )
            )
        traces_3d.append(
            go.Scatter3d(
                x=pos_np[0, :], y=pos_np[1, :], z=pos_np[2, :],
                mode='lines+markers', name='Estimation', legendgroup='Estimation', showlegend=True,
                line=dict(color=colors['filtered'], width=1),
                marker=dict(size=2)
            )
        )

        traces_3d.append(
            go.Scatter3d(
                x=pos_smth_np[0, :], y=pos_smth_np[1, :], z=pos_smth_np[2, :],
                mode='lines+markers', name='Smoothed estimation', legendgroup='Smoothed estimation', showlegend=True,
                line=dict(color=colors['smoothed'], width=1),
                marker=dict(size=2)
            )
        )

        for trace in traces_2d:
            fig.add_trace(trace, row=1, col=1)
        for trace in traces_3d:
            fig.add_trace(trace, row=1, col=2)

        fig.update_layout(
            title=title or (
                f'Trajectory Visualization (aligned with first {n_used} static points)'
                if n_used > 0 else
                'Trajectory Visualization'
            ),
            # width=1000,
            # height=500,
            scene=dict(aspectmode="data")
        )

        axis_titles = ['x [m]', 'y [m]', 'z [m]']
        fig.update_xaxes(title_text=axis_titles[axis2D[0]], row=1, col=1,
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
        fig.update_yaxes(title_text=axis_titles[axis2D[1]], row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )

        fig.update_scenes(
            row=1, col=2,
            xaxis_title='x [m]',
            yaxis_title='y [m]',
            zaxis_title='z [m]'
        )
        if showfig:
            fig.show()

        return fig

    
    def plot_attitude(self, results, title=None, showfig=True):
        """Plot Euler angles"""
        att = results['ATT'].T
        att_unwrapped = xp.as_numpy(xp.unwrap(att) * 180 / xp.pi)
        
        fig = go.Figure()

        labels = ['Roll', 'Pitch', 'Yaw']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    y=att_unwrapped[:, i],
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i], width=2)
                )
            )
        
        fig.update_layout(
            title=title or 'Attitude (Euler Angles)',
            xaxis_title='Time [s]',
            yaxis_title='Angle [deg]',
            template='plotly_white',
            width=600,
            height=300
        )
        if showfig:
            fig.show()

        return fig

    def plot_quaternion(self, results, title=None, showfig=True):
        """Plot quaternion components"""
        quat = xp.as_numpy(results['QUA'].T)
        
        fig = go.Figure()

        labels = ['q0', 'q1', 'q2', 'q3']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i in range(4):
            fig.add_trace(
                go.Scatter(
                    y=quat[:, i],
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i], width=2)
                )
            )
        
        fig.update_layout(
            title=title or 'Orientation',
            template='plotly_white',
            height=300
        )
        if showfig:
            fig.show()

        return fig

    def plot_orientation(self, results, title=None, showfig=True):
        fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Quaternion', 'Attitude'),
                        specs=[[{'type': 'xy'}, {'type': 'xy'}]]
                    )

        """Plot Euler angles"""
        att = results['ATT'].T
        att_unwrapped = xp.as_numpy(xp.unwrap(att) * 180 / xp.pi)
        
        
        labels = ['Roll', 'Pitch', 'Yaw']
        
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    y=att_unwrapped[:, i],
                    mode='lines',
                    name=labels[i],
                    line=dict(width=2)
                ),
                row=1, col=1
            )

        """Plot quaternion components"""
        quat = xp.as_numpy(results['QUA'].T)

        labels = ['qw', 'qx', 'qy', 'qz']
        
        for i in range(4):
            fig.add_trace(
                go.Scatter(
                    y=quat[:, i],
                    mode='lines',
                    name=labels[i],
                    line=dict(width=2)
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=title or 'Orientation',
            template='plotly_white',
            height=300
        )
        if showfig:
            fig.show()

        return fig

    # Helper property methods (to be defined by subclasses)
    def position(self):
        """Return position from state vector"""
        return self.m[self.POS]
    
    def velocity(self):
        """Return velocity from state vector"""
        return self.m[self.VEL]
    
    def orientation(self):
        """Return orientation from state vector"""
        if hasattr(self, 'QUA'):
            return self.m[self.QUA]
        else:
            return self.m[self.ATT]
    
    def bias_gyroscope_additive(self):
        """Return gyroscope bias from state vector"""
        return self.m[self.BGA]
    
    def bias_accelerometer_additive(self):
        """Return accelerometer bias from state vector"""
        return self.m[self.BAA]
    
    def bias_accelerometer_transform(self):
        """Return accelerometer transform bias from state vector"""
        return self.m[self.BAT]
    
    def speed(self):
        """Return speed (magnitude of velocity)"""
        return xp.sqrt(xp.sum(self.m[self.VEL]**2))
    
    def horizontal_speed(self):
        """Return horizontal speed (2D velocity magnitude)"""
        return xp.sqrt(xp.sum(self.m[self.VEL[0:2]]**2))
    
    def get_state(self):
        """Return full state vector"""
        return self.m
    
    def get_state_covariance(self):
        """Return state covariance matrix"""
        return self.P
    
    def get_state_dim(self):
        """Return state dimension"""
        return self.stateDim
