"""
State Space Model - HybVIO
Hybrid Visual-Inertial Odometry implementation
"""

import xp
from generic_filter_smoother import ProblemInterface
from sins import SINS, IMU
from prepare_my_data import calibrate_my_imu

class SSM_HybVIO(ProblemInterface, SINS):
    """
    State Space Model for HybVIO
    
    State vector:
    - Position (POS): [p0, p1, p2]
    - Velocity (VEL): [v0, v1, v2]
    - Orientation (QUA): [qw, qx, qy, qz]
    - Gyro Bias Additive (BGA): [bga0, bga1, bga2]
    - Acc Bias Additive (BAA): [baa0, baa1, baa2]
    - Acc Bias Transform (BAT): [bat0, bat1, bat2]
    """
    
    def __init__(self, Ts:float, imu_uncal:IMU, init_pos=xp.array([[0],[0],[0]]), init_yaw:float=0, still:int=20, **hybvio_kwargs):
        """
        Initialize SSM_HybVIO
        
        Args:
            Ts: Sampling period
            init_pos: Initial position [3x1]
            acc_for_init_orientation: Accelerometer data for initial orientation
            init_yaw: Initial yaw angle
        """
        SINS.__init__(self, Ts)

        self.n_frame = 'ENU'
        self.gravity = xp.array([[0],[0],[-self.g]])
        
        # State indices
        self.POS = slice(0, 3)
        self.VEL = slice(3, 6)
        self.QUA = slice(6, 10)
        self.BGA = slice(10, 13)
        self.BAA = slice(13, 16)
        self.BAT = slice(16, 19)
        
        self.POSE_DIM = 7
        self.INER_DIM = 19
        self.stateDim = 19
        
        # State part names and sizes
        self.STATE_PART_NAMES = ["POS", "VEL", "QUA", "BGA", "BAA", "BAT"]
        self.STATE_PART_SIZES = [3, 3, 4, 3, 3, 3]
        
        # Process noise structure
        self.Q_ACC = slice(0, 3)
        self.Q_GYRO = slice(3, 6)
        self.Q_BGA_DRIFT = slice(6, 9)
        self.Q_BAA_DRIFT = slice(9, 12)
        self.Q_DIM = 12
        
        
        # Initial noise settings
        self.noiseInitialPos = 1e-5
        self.noiseInitialVel = 1e-5
        self.noiseInitialOri = xp.pi / 180 * 0.1
        self.noiseInitialBGA = 0.3 * xp.pi / 180
        self.noiseInitialBAA = 0.3
        self.noiseInitialBAT = 1e-5
        
        # Process noise settings
        self.noiseProcessAcc = 20 #0.5
        self.noiseProcessGyro = 7.5 # 0.5 * xp.pi / 180
        self.noiseProcessBAA = 0.0000001
        self.noiseProcessBGA = 0.0000001 * xp.pi / 180
        self.noiseProcessBAARev = 0.1
        self.noiseProcessBGARev = 0.1
        
        # Initialize state and covariance    
        self.m = xp.zeros((self.stateDim, 1)) 
        self.m[self.POS, :] = xp.array(init_pos).reshape(3, 1)
        self.m[self.QUA, :] = xp.array([1, 0, 0, 0]).reshape(4, 1)
        self.m[self.BAT, :] = xp.ones((3, 1))

        _allowed = {
            'noiseInitialPos', 'noiseInitialVel', 'noiseInitialOri',
            'noiseInitialBGA', 'noiseInitialBAA', 'noiseInitialBAT',
            'noiseProcessAcc', 'noiseProcessGyro',
            'noiseProcessBAA', 'noiseProcessBGA',
            'noiseProcessBAARev', 'noiseProcessBGARev',
        }
        for k, v in hybvio_kwargs.items():
            if k not in _allowed:
                raise KeyError(f"Unknown SSM_HybVIO parameter '{k}'")
            setattr(self, k, v)

        self.set_state_covariance()
        self.set_fixed_process_noise_covariance()

        # ========== Calibrate IMU ==========
        imu, bG = calibrate_my_imu(imu_uncal, still=still, g=self.g)
        self.u_all = xp.vstack((imu.acc,imu.gyr))
        print(f"IMU calibrated. Gyro bias computed as {bG.reshape(3,)/xp.pi*180} deg/s")
        
        # Initial alignment
        acc_for_init_orientation=imu.acc[:, :still]
        self.qua0, self.att0 = self.init_attitude_from_acc(acc_for_init_orientation, init_yaw)
        # q_dict = self.quat_from_two_vectors(self.gravity.reshape(3,), xp.mean(acc_for_init_orientation,axis=1).reshape(3,))
        # self.qua0 = xp.hstack([q_dict['w'],q_dict['vec']]).reshape(4,1)
        # self.att0 = self.qua2att(self.qua0)

        self.m[self.QUA, :] = xp.copy(self.qua0)
        
        # Store initial values (use xp.copy to be backend-agnostic)
        self.m0 = xp.copy(self.m)
        self.P0 = xp.copy(self.P)
        self.Q0 = xp.copy(self.Q)

        # ================
        # self.lockBiases()

        # --- other constant parts
        # (do not need gradients tracking when using AD)
        with xp.no_grad():
            self.Fx_base = xp.eye(self.stateDim,self.stateDim)
            self.Fx_base[self.POS, self.VEL] = self.I3 * self.Ts
            
            self.Fi_base = xp.zeros((self.stateDim, self.Q_DIM))
            self.Fi_base[self.BGA, self.Q_BGA_DRIFT] = self.I3
            self.Fi_base[self.BAA, self.Q_BAA_DRIFT] = self.I3

            self.dS0 = xp.array([[0, self.Ts/2, 0, 0], [-self.Ts/2, 0, 0, 0], 
                                [0, 0, 0, self.Ts/2], [0, 0, -self.Ts/2, 0]])
            self.dS1 = xp.array([[0, 0, self.Ts/2, 0], [0, 0, 0, -self.Ts/2], 
                                [-self.Ts/2, 0, 0, 0], [0, self.Ts/2, 0, 0]])
            self.dS2 = xp.array([[0, 0, 0, self.Ts/2], [0, 0, self.Ts/2, 0], 
                                [0, -self.Ts/2, 0, 0], [-self.Ts/2, 0, 0, 0]])        


    def set_state_covariance(self, **kwargs):
        # Override only the provided parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Initialize covariance       
        self.P = xp.zeros((self.stateDim, self.stateDim))
        self.P[self.POS, self.POS] = self.I3 * self.noiseInitialPos**2
        self.P[self.VEL, self.VEL] = self.I3 * self.noiseInitialVel**2
        self.P[self.QUA, self.QUA] = xp.eye(4) * self.noiseInitialOri**2
        self.P[self.BGA, self.BGA] = self.I3 * self.noiseInitialBGA**2
        self.P[self.BAA, self.BAA] = self.I3 * self.noiseInitialBAA**2
        self.P[self.BAT, self.BAT] = self.I3 * self.noiseInitialBAT**2
    
    def set_fixed_process_noise_covariance(self, **kwargs):
        # Process noise covariance
        # Override only the provided parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Unknown noise parameter '{key}'")

        # Create Q fresh ONCE
        _Q = xp.zeros((self.Q_DIM, self.Q_DIM))
        # self.tag("Q fresh", Q)

        _Q[self.Q_ACC, self.Q_ACC] = self.I3 * self.noiseProcessAcc**2
        _Q[self.Q_GYRO, self.Q_GYRO] = self.I3 * self.noiseProcessGyro**2

        self.Q = _Q

    
    def transition_model(self):
        """
        State transition model (prediction step)
        Implements the prediction equations for position, velocity, and orientation
        Autograd-safe, backend-agnostic transition model.
        Works for both numpy and torch without breaking graph.
        """
        dt = self.Ts

        # Current state
        p = xp.copy(self.m[self.POS,:])
        v = xp.copy(self.m[self.VEL,:])
        q = xp.copy(self.quat)

        # Corrected measurements
        ba = xp.copy(self.m[self.BAA, :])
        Txab = xp.diag(self.m[self.BAT, :].flatten()) @ self.u['acc'] - ba 

        bg = xp.copy(self.m[self.BGA, :])
        w = xp.flatten(self.u['gyro'] - bg)

        _Q = xp.copy(self.Q)
        # Mean reverting random walk for biases
        if self.noiseProcessBAA > 0:
            noise_scale = getattr(self, 'noiseScale', 1.0)
            Q_baa = self.I3 * (noise_scale * self.noiseProcessBAA**2)
            _Q[self.Q_BAA_DRIFT, self.Q_BAA_DRIFT] = Q_baa
            if self.noiseProcessBAARev > 0:
                _Q[self.Q_BAA_DRIFT, self.Q_BAA_DRIFT] = Q_baa * (1 - xp.exp(-2 * dt * self.noiseProcessBAARev)) / (2 * self.noiseProcessBAARev)
        
        if self.noiseProcessBGA > 0:
            noise_scale = getattr(self, 'noiseScale', 1.0)
            Q_bga = self.I3 * (noise_scale * self.noiseProcessBGA**2)
            _Q[self.Q_BGA_DRIFT, self.Q_BGA_DRIFT] = Q_bga
            if self.noiseProcessBGARev > 0:
                _Q[self.Q_BGA_DRIFT, self.Q_BGA_DRIFT] = Q_bga * (1 - xp.exp(-2 * dt * self.noiseProcessBGARev)) / (2 * self.noiseProcessBGARev)
        self.Q = xp.copy(_Q)
        
        # Gyro rotation        
        S = xp.array([
            [0,    -w[0], -w[1], -w[2]],
            [w[0],    0,  -w[2],  w[1]],
            [w[1],  w[2],    0,  -w[0]],
            [w[2], -w[1],  w[0],    0]
        ])
        A = xp.expm(S * (-dt / 2))

        # ================ state update ===========================
        m_new = xp.copy(self.m)
        
        # Rotation
        prev_quat = xp.copy(self.m[self.QUA,:])
        q_new = A @ prev_quat
        C, dC = self.quat2rmat_d(q_new)
        
        m_new[self.POS, :] = p + v * dt  
        m_new[self.VEL, :] = v + (C.T @ Txab + self.gravity) * dt
        m_new[self.QUA, :] = xp.copy(q_new)
        m_new[self.BAT, :] = self.m[self.BAT, :]
        
        # Bias mean reversion
        if self.noiseProcessBAA > 0: #and self.noiseProcessBAARev > 0:
            m_new[self.BAA, :] = ba * xp.exp(-dt * self.noiseProcessBAARev)
        if self.noiseProcessBGA > 0: # and self.noiseProcessBGARev > 0:
            m_new[self.BGA, :] = bg * xp.exp(-dt * self.noiseProcessBGARev)

        self.m = xp.copy(m_new)
        self.quat = xp.copy(q_new)
        
        # ================ Jacobians =============================
        
        
        # Derivatives of velocity w.r.t. quaternion
        blk_vq = xp.zeros((3,4))
        for i in range(4):
            blk_vq[:,i] = xp.flatten(dC[i].T @ Txab * dt)
        
        # Derivatives of quaternion w.r.t. gyroscope noise
        

        blk_qw = xp.hstack([A @ self.dS0 @ prev_quat, A @ self.dS1 @ prev_quat, A @ self.dS2 @ prev_quat])

        _F = xp.copy(self.Fx_base)
        _F[self.VEL, self.QUA] = blk_vq @ A
        _F[self.VEL, self.BGA] = -blk_vq @ A @ blk_qw
        _F[self.VEL, self.BAA] = -C.T * dt
        _F[self.VEL, self.BAT] = C.T @ xp.diag(self.u['acc'].flatten()) * dt
        _F[self.QUA, self.QUA] = A
        _F[self.QUA, self.BGA] = -blk_qw
        _F[self.BGA, self.BGA] = self.I3
        _F[self.BAA, self.BAA] = self.I3

        _G = xp.copy(self.Fi_base)
        _G[self.VEL, self.Q_ACC] = C.T * dt
        _G[self.VEL, self.Q_GYRO] = blk_vq @ A @ blk_qw
        _G[self.QUA, self.Q_GYRO] = xp.copy(blk_qw)

        self.dfdx = _F
        self.dfdq = _G
    
    def measurement_model(self):
        """
        Measurement model
        
        Returns:
            measure_pre: Predicted measurement
        """
        measure_type = self.measureType.lower()
        
        if measure_type == 'zupt':
            # Zero velocity update
            measure_pre = xp.copy(self.m[self.VEL, :])
     
        elif measure_type == 'zrupt':
            # Zero rotation update
            measure_pre = self.m[self.BGA, :]

        elif measure_type == 'pseudohorispeed':
            # Velocity pseudo update (horizontal speed only)
            vxy = self.m[self.VEL, :][0:2].flatten()
            measure_pre = xp.norm(vxy)
            self.dhdx = xp.zeros((1, self.stateDim))
            if measure_pre <= 1e-7:
                return measure_pre
            for i in range(2):
                self.dhdx[0, self.VEL.start + i] = self.m[self.VEL.start + i, 0] / measure_pre
            
        elif measure_type == 'position':
            measure_pre = xp.copy(self.m[self.POS,:])

        elif measure_type == 'zeroheight':
            # Zero height update
            measure_pre = self.m[self.POS.stop - 1, :]
            
        elif measure_type == 'orientation':
            measure_pre = self.m[self.QUA,:]
            
        elif measure_type == 'positionx':
            measure_pre = self.m[self.POS.start, :]
            
        elif measure_type == 'positiony':
            measure_pre = self.m[self.POS.start + 1, :]
        
        else:
            raise ValueError(f"Unknown measurement type: {measure_type}")
        
        return measure_pre
    
    def set_measurement_fixed_part(self, measure_type):
        # Note: current version require define only one measurement type 
        # measure_type = self.measureType.lower()
        self.measureType = measure_type
    
        if measure_type == 'zupt':
            # Zero velocity update
            _H = xp.zeros((3, self.stateDim))
            _H[:, self.VEL] = self.I3
            self.R = self.I3 * self.zuptR
            self.dhdr = self.I3
            self.dhdx = _H
            
        elif measure_type == 'zrupt':
            # Zero rotation update
            self.dhdx = xp.zeros((3, self.stateDim))
            self.dhdx[:, self.BGA] = self.I3
            self.R = self.I3 * self.rotationZuptR
            self.dhdr = self.I3

        elif measure_type == 'pseudohorispeed':
            # Velocity pseudo update (horizontal speed only)
            self.R = xp.eye(1) * self.pseudoVelocityR
            self.dhdr = xp.array([[1]])
            
        elif measure_type == 'position':
            self.R = self.I3 * self.noiseInitialPos
            _H = xp.zeros((3, self.stateDim))
            _H[:, self.POS] = self.I3
            self.dhdr = self.I3
            self.dhdx = _H
            
        elif measure_type == 'zeroheight':
            # Zero height update
            self.dhdx = xp.zeros((1, self.stateDim))
            self.dhdx[0, self.POS.stop - 1] = 1
            self.R = xp.eye(1) * self.noiseInitialPos
            self.dhdr = xp.array([[1]])
            
        elif measure_type == 'orientation':
            self.dhdx = xp.zeros((4, self.stateDim))
            self.dhdx[:, self.QUA] = xp.eye(4)
            self.R = xp.eye(4) * self.noiseInitialOri
            self.dhdr = xp.eye(4)
            
        elif measure_type == 'positionx':
            self.R = self.noiseInitialPos
            self.dhdx = xp.zeros((1, self.stateDim))
            self.dhdx[0, self.POS.start] = 1
            self.dhdr = xp.array([[1]])
            
        elif measure_type == 'positiony':
            self.R = self.noiseInitialPos
            self.dhdx = xp.zeros((1, self.stateDim))
            self.dhdx[0, self.POS.start + 1] = 1
            self.dhdr = xp.array([[1]])
        
        else:
            raise ValueError(f"Unknown measurement type: {measure_type}")