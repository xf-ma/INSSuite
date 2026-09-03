"""
State Space Model - Sola
Error-state Kalman filter following Sola's formulation
"""

import xp
from generic_filter_smoother import ProblemInterface
from sins import SINS, IMU
from prepare_my_data import calibrate_my_imu

class SSM_Sola(ProblemInterface, SINS):
    """
    State Space Model following Sola's error-state formulation
    
    State vector:
    - Position (POS): [p0, p1, p2]
    - Velocity (VEL): [v0, v1, v2]
    - Attitude (ATT): [att0, att1, att2] (error angles)
    - Acc Bias Additive (BAA): [baa0, baa1, baa2]
    - Gyro Bias Additive (BGA): [bga0, bga1, bga2]
    
    Note: Quaternion is stored separately (not in state vector for ESKF)
    """
    
    def __init__(self, Ts:float, imu_uncal:IMU, init_pos=xp.array([[0],[0],[0]]), init_yaw:float=0, still:int=20, **sola_kwargs):
        """
        Initialize SSM_Sola
        
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
        self.ATT = slice(6, 9)
        self.BAA = slice(9, 12)
        self.BGA = slice(12, 15)
        
        # Simplified Sola model: gravity is constant (not estimated as state)
        self.INER_DIM = 15
        self.stateDim = 15
        
        # State part names and sizes
        self.STATE_PART_NAMES = ["POS", "VEL", "ATT", "BAA", "BGA"]
        self.STATE_PART_SIZES = [3, 3, 3, 3, 3]
        
        # Process noise structure
        self.Q_ACC = slice(0, 3)
        self.Q_GYRO = slice(3, 6)
        self.Q_BAA_DRIFT = slice(6, 9)
        self.Q_BGA_DRIFT = slice(9, 12)
        self.Q_DIM = 12      
        
        # Initial noise settings (Sola's values)
        self.noiseInitialPos = 1e-5
        self.noiseInitialVel = 1e-5
        self.noiseInitialOri = xp.pi / 180 * 0.1
        self.noiseInitialBGA = 0.3 * xp.pi / 180
        self.noiseInitialBAA = 0.3
        
        # Process noise settings (Sola's values)
        self.noiseProcessAcc = 5 
        self.noiseProcessGyro = 50 * xp.pi / 180
        self.noiseProcessBAA = 1e-2
        self.noiseProcessBGA = 1e-2 * xp.pi / 180
        
        # Initialize state and covariance
        self.m = xp.zeros((self.stateDim, 1))       
        self.qua0 = xp.array([[1], [0], [0], [0]])     
        self.m[self.POS] = xp.copy(init_pos)

        _allowed = {
            'noiseInitialPos', 'noiseInitialVel', 'noiseInitialOri',
            'noiseInitialBGA', 'noiseInitialBAA',
            'noiseProcessAcc', 'noiseProcessGyro',
            'noiseProcessBAA', 'noiseProcessBGA',
        }
        for k, v in sola_kwargs.items():
            if k not in _allowed:
                raise KeyError(f"Unknown SSM_Sola parameter '{k}'")
            setattr(self, k, v)

        self.set_state_covariance()
        self.set_fixed_process_noise_covariance()

        # ========== Calibrate IMU ==========
        imu, bG = calibrate_my_imu(imu_uncal, still=still, g=self.g)
        self.u_all = xp.vstack((imu.acc,imu.gyr))
        # print(f"IMU calibrated. Gyro bias computed as {bG.reshape(3,)/xp.pi*180} deg/s")
                
        # Initial alignment
        acc_for_init_orientation=imu.acc[:, :still]
        self.qua0, self.att0 = self.init_attitude_from_acc(acc_for_init_orientation, init_yaw)
        # q_dict = self.quat_from_two_vectors(self.gravity.reshape(3,), xp.mean(acc_for_init_orientation,axis=1).reshape(3,))
        # self.qua0 = xp.hstack([q_dict['w'],q_dict['vec']]).reshape(4,1)
        # self.att0 = self.qua2att(self.qua0)

        self.m[self.ATT] = xp.copy(self.att0)
        self.quat = xp.copy(self.qua0)
        
        # Store initial values
        self.m0 = xp.copy(self.m)
        self.P0 = xp.copy(self.P)
        self.Q0 = xp.copy(self.Q)

        # --- other constant parts
        # (do not need gradients tracking when using AD)
        with xp.no_grad():
            self.Fx_base = xp.eye(self.stateDim, self.stateDim)
            self.Fx_base[self.POS, self.VEL] = self.I3 * self.Ts
            self.Fx_base[self.ATT, self.BGA] = -self.I3 * self.Ts
            
            # Process noise Jacobian
            self.Fi_base = xp.zeros((self.stateDim, self.Q_DIM))
            self.Fi_base[self.VEL, self.Q_ACC] = self.I3
            self.Fi_base[self.ATT, self.Q_GYRO] = self.I3
            self.Fi_base[self.BAA, self.Q_BAA_DRIFT] = self.I3
            self.Fi_base[self.BGA, self.Q_BGA_DRIFT] = self.I3

            Hx = xp.zeros((3, self.stateDim + 1))
            Hx[:, self.VEL] = self.I3
            self._Hx = Hx

    
    def set_state_covariance(self, **kwargs):
        # Override only the provided parameters
        if xp._xp.use_torch:
            print("Now using torch")

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Initialize covariance
        self.P = xp.zeros((self.stateDim, self.stateDim))
        self.P[self.POS, self.POS] = self.I3 * self.noiseInitialPos**2
        self.P[self.VEL, self.VEL] = self.I3 * self.noiseInitialVel**2
        self.P[self.ATT, self.ATT] = self.I3 * self.noiseInitialOri**2
        self.P[self.BGA, self.BGA] = self.I3 * self.noiseInitialBGA**2
        self.P[self.BAA, self.BAA] = self.I3 * self.noiseInitialBAA**2
               
    def set_fixed_process_noise_covariance(self, **kwargs):
        """
        kwargs: noiseProcessAcc, noiseProcessGyro, noiseProcessBAA, noiseProcessBGA, etc.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 
                # self.tag(key, value)
            else:
                raise KeyError(f"Unknown noise parameter '{key}'")

        dt = self.Ts

        # Create Q fresh ONCE
        Q = xp.zeros((self.Q_DIM, self.Q_DIM))#, device=device)

        # ---- Sola's formulation ----
        Vi      = (self.noiseProcessAcc**2)  * self.I3 * dt**2
        THETAi  = (self.noiseProcessGyro**2) * self.I3 * dt**2
        Ai      = (self.noiseProcessBAA**2)  * self.I3 * dt
        OMEGAi  = (self.noiseProcessBGA**2)  * self.I3 * dt

        Q[self.Q_ACC,           self.Q_ACC]           = Vi
        Q[self.Q_GYRO,          self.Q_GYRO]          = THETAi
        Q[self.Q_BAA_DRIFT,     self.Q_BAA_DRIFT]     = Ai
        Q[self.Q_BGA_DRIFT,     self.Q_BGA_DRIFT]     = OMEGAi

        self.Q = Q

    def transition_model(self):
        """
        State transition model (Sola's error-state formulation)
        
        This implements the error-state dynamics where the nominal state
        is propagated separately from the error state
        """

        dt = self.Ts
        
        # Corrected measurements
        a_h = self.u['acc'] - self.m[self.BAA]
        omega_h = self.u['gyro'] - self.m[self.BGA]
        
        # Current state
        p = self.m[self.POS,:]
        v = self.m[self.VEL,:]
        q = self.quat
        
        # Rotation matrix from quaternion
        C = self.qua2dcm(q, 'Cnb')

        # Quaternion integration
        q_incre = self.rotation_vec2quaternion(omega_h * dt)
        self.quat = self.quaternion_product(q, q_incre)

        # Nominal state propagation
        # (Avoid modifying self.m directly to prevent autograd issues)
        m_new = xp.copy(self.m)
        m_new[self.POS] = p + v * dt + 0.5 * (C @ a_h + self.gravity) * dt**2
        m_new[self.VEL] = v + (C @ a_h + self.gravity) * dt
        m_new[self.ATT] = self.qua2att(self.quat)
        self.m = m_new  

        
        # Error-state Jacobian matrices
        
        # Rotation matrix for small angle approximation
        R_wdt = self.rotation_vec2rotation_matrix(omega_h * dt)
        
        # Skew-symmetric matrix of corrected acceleration
        S = self.skew(a_h)
        
        # State transition matrix (error state)
        _Fx = xp.copy(self.Fx_base)
        _Fx[self.VEL, self.ATT] = -C @ S * dt
        _Fx[self.VEL, self.BAA] = -C * dt
        _Fx[self.ATT, self.ATT] = R_wdt.T
        
        self.dfdx = _Fx
        self.dfdq = self.Fi_base
    
    def set_measurement_fixed_part(self, measure_type):
        # Note: current version require define only one measurement type 
        """
        Measurement model
        
        For ESKF, need to account for quaternion parameterization
        """
        # measure_type = self.measureType.lower()
        self.measureType = measure_type
        
        if measure_type == 'zupt':
            # Zero velocity update
            self.R = self.I3 * self.zuptR
            self.dhdr = self.I3         
               
            
        elif measure_type == 'position':
            self.R = self.I3 * self.noiseInitialPos
            _H = xp.zeros((3, self.stateDim))
            _H[:, self.POS] = self.I3
            self.dhdx = _H
            self.dhdr = self.I3
        
        else:
            raise ValueError(f"Unknown measurement type: {measure_type}")
        
    

    def measurement_model(self):
        """
        Measurement model
        
        For ESKF, need to account for quaternion parameterization
        """
        measure_type = self.measureType.lower()
        
        if measure_type == 'zupt':
            # Zero velocity update
           
            measure_pre = xp.copy(self.m[self.VEL])
            
            # Quaternion to error angle Jacobian
            q = self.quat.flatten()
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            
            Qdtheta = 0.5 * xp.array([
                [-qx, -qy, -qz],
                [qw, -qz, qy],
                [qz, qw, -qx],
                [-qy, qx, qw]
            ])
            
            # Transformation from error state to nominal state
            Xdx = xp.block_diag(xp.eye(6), Qdtheta, xp.eye(6))
            self.dhdx = self._Hx @ Xdx
            
        elif measure_type == 'position':
            measure_pre = xp.copy(self.m[self.POS])
        
        else:
            raise ValueError(f"Unknown measurement type: {measure_type}")
        
        return measure_pre
    
    def comp_internal_states(self, dx):
        """
        Compose error state with nominal state (ESKF)
        
        This is the key step in error-state filtering where we
        inject the error state correction into the nominal state
        
        Args:
            dx: Error state correction
        """
        epsilon = dx[self.ATT]
        # Update quaternion (compose with error rotation)
        delta_q = self.rotation_vec2quaternion(epsilon)

        x_old = xp.copy(self.m).reshape(-1, 1)
        q_old = xp.copy(self.quat).reshape(-1, 1)
        self.quat = self.quaternion_product(q_old, delta_q)

        # Update nominal state
        m_new = x_old + dx.reshape(-1, 1)      
        m_new[self.ATT] = self.qua2att(self.quat)
        self.m = m_new
