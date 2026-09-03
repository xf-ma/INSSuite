"""
State Space Model - OpenShoe
Foot-mounted inertial navigation system implementation
"""

import xp
from generic_filter_smoother import ProblemInterface
from sins import SINS, IMU
from prepare_my_data import calibrate_my_imu

class SSM_OpenShoe(ProblemInterface, SINS):
    """
    State Space Model for OpenShoe
    
    State vector (depending on configuration):
    - Position (POS): [p0, p1, p2]
    - Velocity (VEL): [v0, v1, v2]
    - Attitude (ATT): [roll, pitch, yaw]
    - Bias Acc Additive (BAA): [baa0, baa1, baa2] (optional)
    - Bias Gyro Additive (BGA): [bga0, bga1, bga2] (optional)
    - Acc Scale Factor (ASF): [asf0, asf1, asf2] (optional)
    - Gyro Scale Factor (GSF): [gsf0, gsf1, gsf2] (optional)
    """
    
    def __init__(self, Ts:float, imu_uncal:IMU, init_pos=xp.array([[0],[0],[0]]), init_yaw:float=0, still:int=20, biases_switch='on', scalefactors_switch='off', **openshoe_kwargs):
        """
        Initialize SSM_OpenShoe
        
        Args:
            Ts: Sampling period
            init_pos: Initial position [3x1]
            acc_for_init_orientation: Accelerometer data for initial orientation
            init_yaw: Initial yaw angle
            biases_switch: 'on' or 'off' for bias estimation
            scalefactors_switch: 'on' or 'off' for scale factor estimation
        """
        SINS.__init__(self, Ts)

        self.n_frame = 'NED'
        self.gravity = xp.array([[0],[0],[self.g]])
        
        # Basic state indices
        self.POS = slice(0, 3)
        self.VEL = slice(3, 6)
        self.ATT = slice(6, 9)
        self.POSE_DIM = 6
        
        self.biases = biases_switch
        self.scalefactors = scalefactors_switch
        
        
        # Initial noise settings
        self.noiseInitialPos = 1e-5
        self.noiseInitialVel = 1e-5
        self.noiseInitialAtt = xp.pi / 180 * 0.1
        self.noiseInitialBAA = 0.3
        self.noiseInitialBGA = 0.3 * xp.pi / 180
        self.noiseInitialASF = 0.0001
        self.noiseInitialGSF = 0.00001
        
        # Process noise settings
        self.noiseProcessAcc = 0.5
        self.noiseProcessGyro = 0.5 * xp.pi / 180
        self.noiseProcessBAA = 0.1#0.0000001
        self.noiseProcessBGA = 0.1#0.0000001 * xp.pi / 180
        
        # Bias instability time constants
        self.noiseProcessBAARev = xp.inf()
        self.noiseProcessBGARev = xp.inf()
        
        # Process noise indices
        self.Q_ACC = slice(0, 3)
        self.Q_GYRO = slice(3, 6)
        self.Q_BAA_DRIFT = slice(6, 9)
        self.Q_BGA_DRIFT = slice(9, 12)

        _allowed = {
            'noiseInitialPos', 'noiseInitialVel', 'noiseInitialAtt',
            'noiseInitialBAA', 'noiseInitialBGA', 'noiseInitialASF', 'noiseInitialGSF',
            'noiseProcessAcc', 'noiseProcessGyro',
            'noiseProcessBAA', 'noiseProcessBGA',
            'noiseProcessBAARev', 'noiseProcessBGARev',
        }
        for k, v in openshoe_kwargs.items():
            if k not in _allowed:
                raise KeyError(f"Unknown SSM_OpenShoe parameter '{k}'")
            setattr(self, k, v)

        # Configure state space based on switches
        self._configure_state_space()
        self.set_state_covariance()
        self.set_fixed_process_noise_covariance()

        # Initialize state and covariance
        self.m = xp.zeros((self.stateDim, 1))
        self.m[self.POS] = xp.copy(init_pos)
        
        # ========== Calibrate IMU ==========
        imu, bG = calibrate_my_imu(imu_uncal, still=still, g=self.g)
        self.u_all = xp.vstack((imu.acc,imu.gyr))
        print(f"IMU calibrated. Gyro bias computed as {bG.reshape(3,)/xp.pi*180} deg/s")
        
        # Initial alignment
        acc_for_init_orientation=imu.acc[:, :still]
        f_u = xp.mean(acc_for_init_orientation[0,:])
        f_v = xp.mean(acc_for_init_orientation[1,:])
        f_w = xp.mean(acc_for_init_orientation[2,:])
        roll = xp.arctan2(-f_v, -f_w)
        pitch = xp.arctan2(f_u, xp.sqrt(f_v**2 + f_w**2))
        attitude = xp.array([roll,pitch,init_yaw])
        print(f'----- Initial attitude (deg): {attitude/xp.pi*180} -----')

        # Calculate quaternion
        Rb2t = self.Rt2b(attitude).T
        self.quat = self.dcm2q(Rb2t).reshape(-1,1)
        self.qua0 = xp.copy(self.quat)

        self.att0 = xp.array([
            xp.arcsin(f_v / f_w).reshape(1,),
            xp.arctan2(-f_u, abs(self.gravity[2])).reshape(1,),
            xp.array(init_yaw).reshape(1,)
        ])
        self.m[self.ATT] = self.att0
        print(f'----- Initial attitude (deg): {self.att0/xp.pi*180} -----')
        
        # Store initial values
        self.m0 = xp.copy(self.m).reshape(-1, 1)
        self.P0 = xp.copy(self.P)
        self.Q0 = xp.copy(self.Q)

        # --- other constant parts
        # (do not need gradients tracking when using AD)
        with xp.no_grad():
            self.A = xp.eye(6)
            self.A[0, 3] = self.Ts
            self.A[1, 4] = self.Ts
            self.A[2, 5] = self.Ts
            
            self.B = xp.vstack([
                (self.Ts**2 / 2) * xp.eye(3),
                self.Ts * xp.eye(3)
            ])

            # Configure constant parts of F and G based on error model
            self.Fc_base = xp.zeros((self.stateDim, self.stateDim))
            self.Fc_base[self.POS, self.VEL] = self.I3

            self.Gc_base = xp.zeros((self.stateDim, self.Q_DIM))
            if self.biases == 'on':
                self.Gc_base[self.BAA, self.Q_BAA_DRIFT] = self.I3
                self.Gc_base[self.BGA, self.Q_BGA_DRIFT] = self.I3
            else:
                pass


    def set_state_covariance(self, **kwargs):
        # Override only the provided parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if self.scalefactors == 'on' and self.biases == 'on':
            # Both scale and bias errors         
            self.P = xp.zeros((self.stateDim, self.stateDim))
            self.P[self.BAA, self.BAA] = xp.eye(3) * self.noiseInitialBAA**2
            self.P[self.BGA, self.BGA] = xp.eye(3) * self.noiseInitialBGA**2
            self.P[self.ASF, self.ASF] = xp.eye(3) *self.noiseInitialASF**2
            self.P[self.GSF, self.GSF] = xp.eye(3) *self.noiseInitialGSF**2

            
        elif self.scalefactors == 'on' and self.biases == 'off':
            # Scale errors only           
            self.P = xp.zeros((self.stateDim, self.stateDim))
            self.P[self.ASF, self.ASF] = xp.eye(3) * self.noiseInitialASF**2
            self.P[self.GSF, self.GSF] = xp.eye(3) * self.noiseInitialGSF**2
            
            
        elif self.scalefactors == 'off' and self.biases == 'on':
            # Bias errors only           
            self.P = xp.zeros((self.stateDim, self.stateDim))
            self.P[self.BAA, self.BAA] = xp.eye(3) * self.noiseInitialBAA**2
            self.P[self.BGA, self.BGA] = xp.eye(3) * self.noiseInitialBGA**2

            
        else:
            # Standard errors only        
            self.P = xp.zeros((self.stateDim, self.stateDim))

        
        # General P values
        self.P[self.POS, self.POS] = xp.eye(3) * self.noiseInitialPos**2
        self.P[self.VEL, self.VEL] = xp.eye(3) * self.noiseInitialVel**2
        self.P[self.ATT, self.ATT] = xp.eye(3) * self.noiseInitialAtt**2
    
    def set_fixed_process_noise_covariance(self, **kwargs):
        # Override only the provided parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        """Configure state space dimensions based on biases and scale factors"""
        if self.scalefactors == 'on' and self.biases == 'on':
            # Both scale and bias errors
            Q = xp.zeros((self.Q_DIM, self.Q_DIM))
            Q[self.Q_BAA_DRIFT, self.Q_BAA_DRIFT] = xp.eye(3) * self.noiseProcessBAA**2
            Q[self.Q_BGA_DRIFT, self.Q_BGA_DRIFT] = xp.eye(3) * self.noiseProcessBGA**2
            
        elif self.scalefactors == 'on' and self.biases == 'off':
            # Scale errors only
            Q = xp.zeros((self.Q_DIM, self.Q_DIM))
            
        elif self.scalefactors == 'off' and self.biases == 'on':
            # Bias errors only
            Q = xp.zeros((self.Q_DIM, self.Q_DIM))
            Q[self.Q_BAA_DRIFT, self.Q_BAA_DRIFT] = xp.eye(3) * self.noiseProcessBAA**2
            Q[self.Q_BGA_DRIFT, self.Q_BGA_DRIFT] = xp.eye(3) * self.noiseProcessBGA**2
            
        else:
            # Standard errors only
            Q = xp.zeros((self.Q_DIM, self.Q_DIM))
        
        # General Q values
        Q[self.Q_ACC, self.Q_ACC] = xp.eye(3) * self.noiseProcessAcc**2
        Q[self.Q_GYRO, self.Q_GYRO] = xp.eye(3) * self.noiseProcessGyro**2

        self.Q = Q

    
    def _configure_state_space(self):
        """Configure state space dimensions based on biases and scale factors"""
        if self.scalefactors == 'on' and self.biases == 'on':
            # Both scale and bias errors
            self.BAA = slice(9, 12)
            self.BGA = slice(12, 15)
            self.ASF = slice(15, 18)
            self.GSF = slice(18, 21)
            
            self.STATE_PART_NAMES = ["POS", "VEL", "ATT", "BAA", "BGA", "ASF", "GSF"]
            self.STATE_PART_SIZES = [3, 3, 3, 3, 3, 3, 3]
            self.stateDim = 21
            
            self.Q_DIM = 12

            
        elif self.scalefactors == 'on' and self.biases == 'off':
            # Scale errors only
            self.ASF = slice(9, 12)
            self.GSF = slice(12, 15)
            
            self.STATE_PART_NAMES = ["POS", "VEL", "ATT", "ASF", "GSF"]
            self.STATE_PART_SIZES = [3, 3, 3, 3, 3]
            self.stateDim = 15
            
            self.Q_DIM = 6
            
        elif self.scalefactors == 'off' and self.biases == 'on':
            # Bias errors only
            self.BAA = slice(9, 12)
            self.BGA = slice(12, 15)
            
            self.STATE_PART_NAMES = ["POS", "VEL", "ATT", "BAA", "BGA"]
            self.STATE_PART_SIZES = [3, 3, 3, 3, 3]
            self.stateDim = 15
            
            self.Q_DIM = 12
            
        else:
            # Standard errors only
            self.STATE_PART_NAMES = ["POS", "VEL", "ATT"]
            self.STATE_PART_SIZES = [3, 3, 3]
            self.stateDim = 9
            
            self.Q_DIM = 6
        
    
    def transition_model(self):
        """State transition model"""
        # Compensate IMU measurements
        u_h = self.comp_imu_errors(self.u)
        
        # Update navigation states
        self.navigation_equations(u_h)
        
        # Update state transition matrix
        self.dfdx, self.dfdq = self.state_matrix(u_h)
    
    def measurement_model(self):
        """Measurement model"""
        measure_type = self.measureType.lower()
        
        if measure_type == 'zupt':
            measure_pre = self.m[self.VEL]
            
        elif measure_type == 'position':
            measure_pre = self.m[self.POS]
        else:
            raise ValueError(f"Unknown measurement type: {measure_type}")
        
        return measure_pre
    
    def set_measurement_fixed_part(self, measure_type):
        # Note: current version require define only one measurement type 
        # measure_type = self.measureType.lower()

        self.measureType = measure_type
        
        if measure_type == 'zupt':
            self.dhdx = xp.zeros((3, self.stateDim))
            self.dhdx[:, self.VEL] = xp.eye(3)
            self.R = xp.eye(3) * self.zuptR
            self.dhdr = xp.eye(3)
            
        elif measure_type == 'position':
            self.R = xp.eye(3) * self.noiseInitialPos
            self.dhdx = xp.zeros((3, self.stateDim))
            self.dhdx[:, self.POS] = xp.eye(3)
            self.dhdr = xp.eye(3)
        else:
            raise ValueError(f"Unknown measurement type: {measure_type}")
        
    
    def comp_imu_errors(self, u_in):
        """Compensate IMU measurements for biases and scale factors"""
        u_in_vec = xp.concatenate([u_in['acc'], u_in['gyro']])
        
        if self.scalefactors == 'on' and self.biases == 'on':
            temp = 1.0 / (xp.ones(6) - self.m[15:21,:])
            u_out = xp.diag(temp) @ u_in_vec + self.m[9:15]
        elif self.scalefactors == 'on' and self.biases == 'off':
            temp = 1.0 / (xp.ones(6) - self.m[9:15,:])
            u_out = xp.diag(temp) @ u_in_vec
        elif self.scalefactors == 'off' and self.biases == 'on':
            u_out = u_in_vec + self.m[9:15,:]
        else:
            u_out = u_in_vec
        
        return {'acc': u_out[0:3,:], 'gyro': u_out[3:6,:]}
    
    def navigation_equations(self, uh):
        """Mechanized navigation equations"""
        x = xp.copy(self.m).reshape(-1,1)
        q = xp.copy(self.quat).reshape(-1,1)
        y = xp.zeros_like(x)
        dt = self.Ts
        
        # Update quaternion
        w_tb = uh['gyro']
        P = w_tb[0] * dt
        Q = w_tb[1] * dt
        R = w_tb[2] * dt
        OMEGA = xp.zeros((4, 4))
        o = xp.array(0).reshape(1,)
        
        OMEGA[0, :] = 0.5 * xp.array([o, R, -Q, P]).reshape(4,)
        OMEGA[1, :] = 0.5 * xp.array([-R, o, P, Q]).reshape(4,)
        OMEGA[2, :] = 0.5 * xp.array([Q, -P, o, R]).reshape(4,)
        OMEGA[3, :] = 0.5 * xp.array([-P, -Q, -R, o]).reshape(4,)
        
        v = xp.norm(w_tb) * dt
        
        if v != 0:
            q = (xp.cos(v/2) * xp.eye(4) + 2/v * xp.sin(v/2) * OMEGA) @ q
            q = q / xp.norm(q)
        
        # Get attitude from quaternion
        Rb2t = self.q2dcm(q)
        Rrfu = xp.diag(xp.array([1, -1, -1])) @ Rb2t
        
        y[6] = xp.arctan2(Rrfu[2, 1], Rrfu[2, 2])  # pitch
        y[7] = xp.arcsin(-Rrfu[2, 0])  # roll
        y[8] = xp.arctan2(Rrfu[1, 0], Rrfu[0, 0])  # yaw
        
        # Update position and velocity
        g_t = self.gravity
        f_t = self.q2dcm(q) @ uh['acc']
        acc_t = f_t + g_t
        
        y[0:6] = self.A @ x[0:6] + self.B @ acc_t
        
        self.m = xp.reshape(y,(-1,1))
        self.quat = xp.reshape(q,(-1,1))
    
    def state_matrix(self, uh):
        """Calculate state transition and process noise gain matrices"""
        dt = self.Ts
        Rb2t = self.q2dcm(self.quat)
        f_t = (Rb2t @ uh['acc']).reshape(3,)

        # Skew symmetric matrix
        St = xp.array([
            [0, -f_t[2], f_t[1]],
            [f_t[2], 0, -f_t[0]],
            [-f_t[1], f_t[0], 0]
        ])
        
        Da = xp.diag(uh['acc'])
        Dg = xp.diag(uh['gyro'])

        B1 = -1/self.noiseProcessBAARev * xp.eye(3)
        B2 = -1/self.noiseProcessBGARev * xp.eye(3)
        
        # Configure F and G based on error model
        _Fc = xp.copy(self.Fc_base)
        _Fc[self.VEL, self.ATT] = St
        if self.biases == 'on':
            _Fc[self.VEL, self.BAA] = Rb2t
            _Fc[self.ATT, self.BGA] = -Rb2t
            _Fc[self.BAA, self.BAA] = B1
            _Fc[self.BGA, self.BGA] = B2

        if self.scalefactors == 'on':
            _Fc[self.VEL, self.ASF] = Rb2t@Da
            _Fc[self.ATT, self.GSF] = -Rb2t@Dg

        _Gc = xp.copy(self.Gc_base)
        _Gc[self.VEL, self.Q_ACC] = Rb2t
        _Gc[self.ATT, self.Q_GYRO] = -Rb2t

        _F = xp.eye(_Fc.shape[0]) + dt * _Fc
        _G = dt * _Gc
        
        return _F, _G
    
    def comp_internal_states(self, dx):
        """Correct estimated states with Kalman filter perturbations"""
        x_in = xp.copy(self.m).reshape(-1,1)
        q_in = xp.copy(self.quat).reshape(-1,1)
        
        R = self.q2dcm(q_in)
        x_out = x_in + dx.reshape(-1,1)
        
        epsilon = dx[6:9].reshape(3,)
        OMEGA = xp.array([
            [0, -epsilon[2], epsilon[1]],
            [epsilon[2], 0, -epsilon[0]],
            [-epsilon[1], epsilon[0], 0]
        ])
        R = (xp.eye(3) - OMEGA) @ R
        
        x_out[6] = xp.arctan2(R[2, 1], R[2, 2])
        x_out[7] = -xp.arctan(R[2, 0] / xp.sqrt(1 - R[2, 0]**2))
        x_out[8] = xp.arctan2(R[1, 0], R[0, 0])
        
        q_out = self.dcm2q(R)
        
        self.m = x_out.reshape(-1,1)
        self.quat = q_out.reshape(-1,1)
    
    def q2dcm(self, q):
        """Convert quaternion to DCM"""
        p = xp.zeros(6)
        p[0:4] = (q**2).reshape(4,)
        p[4] = p[1] + p[2]
        
        if p[0] + p[3] + p[4] != 0:
            p[5] = 2 / (p[0] + p[3] + p[4])
        else:
            p[5] = 0
        
        R = xp.zeros((3, 3))
        R[0, 0] = 1 - p[5] * p[4]
        R[1, 1] = 1 - p[5] * (p[0] + p[2])
        R[2, 2] = 1 - p[5] * (p[0] + p[1])
        
        p[0] = p[5] * q[0]
        p[1] = p[5] * q[1]
        p[4] = p[5] * q[2] * q[3]
        p[5] = p[0] * q[1]
        
        R[0, 1] = p[5] - p[4]
        R[1, 0] = p[5] + p[4]
        
        p[4] = p[1] * q[3]
        p[5] = p[0] * q[2]
        
        R[0, 2] = p[5] + p[4]
        R[2, 0] = p[5] - p[4]
        
        p[4] = p[0] * q[3]
        p[5] = p[1] * q[2]
        
        R[1, 2] = p[5] - p[4]
        R[2, 1] = p[5] + p[4]
        
        return R
    
    def dcm2q(self, R):
        """Convert DCM to quaternion"""
        T = 1 + R[0, 0] + R[1, 1] + R[2, 2]
        
        if T > 1e-8:
            S = 0.5 / xp.sqrt(T)
            qw = 0.25 / S
            qx = (R[2, 1] - R[1, 2]) * S
            qy = (R[0, 2] - R[2, 0]) * S
            qz = (R[1, 0] - R[0, 1]) * S
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = xp.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = xp.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = xp.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S
        
        return xp.array([[qx], [qy], [qz], [qw]])
    
    def Rt2b(self, ang):
        """Rotation matrix from t to b frame"""
        ang = ang.reshape(3,)
        cr, sr = xp.cos(ang[0]), xp.sin(ang[0])
        cp, sp = xp.cos(ang[1]), xp.sin(ang[1])
        cy, sy = xp.cos(ang[2]), xp.sin(ang[2])
        
        R = xp.array([
            [cy*cp, sy*cp, -sp],
            [-sy*cr + cy*sp*sr, cy*cr + sy*sp*sr, cp*sr],
            [sy*sr + cy*sp*cr, -cy*sr + sy*sp*cr, cp*cr]
        ])
        
        return R