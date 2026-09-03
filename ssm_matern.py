"""
State Space Model - Matern GP Bias
Error-state Kalman filter with IMU biases modeled as Matern 3/2 Gaussian Process
represented via its equivalent stochastic differential equation (SDE).

The Matern 3/2 GP is expressed as the solution of the 2nd-order linear SDE:
    d[b_i; b_dot_i] = A_bi [b_i; b_dot_i] dt + G_bi n_bi(t)
where
    A_bi = I_3 x [[0, 1], [-3/l^2, -2*sqrt(3)/l]]   (Kronecker product)
    G_bi = I_3 x [0; 1]
    q_c  = 4 sigma^2 lambda^3,  lambda = sqrt(3)/l   (spectral density)

The exact discrete-time transition F_bi and process noise covariance Q_delta_bi are
obtained via matrix fraction decomposition (Sarkka & Solin 2019):

    M_bi = [[A_bi,  G_bi qc G_bi^T],  * dt
            [  0,    -A_bi^T       ]]

    exp(M_bi) = [[B, C],   =>  F_bi = B,  Q_delta_bi = C D^{-1} = C F_bi^T
                 [0, D]]

Error state ordering (matches theory):
    delta_x = [delta_p, delta_v, delta_theta,
               delta_b_a_dot, delta_b_a,
               delta_b_omega_dot, delta_b_omega]

References:
    Sarkka & Solin, "Applied Stochastic Differential Equations", 2019.
    Sola et al., "A micro Lie theory for state estimation in robotics", 2018.
"""

import numpy as np
from scipy.linalg import expm as scipy_expm
import xp
from generic_filter_smoother import ProblemInterface
from sins import SINS, IMU
from prepare_my_data import calibrate_my_imu


def _matern32_discrete(lengthscale: float, sigma_sq: float, dt: float):
    """
    Compute discrete-time matrices for a scalar Matern 3/2 GP.

    Uses matrix fraction decomposition as in Sarkka & Solin 2019:
        M_bi = [[A_bi, G qc G^T], [0, -A_bi^T]] * dt
        exp(M_bi) = [[B, C], [0, D]]
        F_bi = B,   Q_d = C @ D^{-1} = C @ F_bi^T

    State ordering: [b; b_dot]  (bias first, rate second — natural SDE order)

    Returns
    -------
    A_d : (2, 2)  discrete transition matrix in [b; b_dot] ordering
    Q_d : (2, 2)  discrete process noise covariance (symmetric PD)
    """
    lam = np.sqrt(3.0) / lengthscale
    q_c = 4.0 * sigma_sq * lam**3

    F_c = np.array([[0.0,      1.0     ],
                    [-lam**2, -2.0*lam ]])
    G   = np.array([[0.0], [1.0]])

    # Matrix fraction decomposition (theory's formulation)
    n = 2
    M = np.zeros((2*n, 2*n))
    M[:n, :n] =  F_c              # A_bi
    M[:n, n:] =  G @ (q_c * G.T) # G * qc * G^T  = [[0,0],[0,qc]]
    M[n:, n:] = -F_c.T            # -A_bi^T
    eMdt = scipy_expm(M * dt)

    B = eMdt[:n, :n]   # F_bi = exp(A_bi * dt)
    C = eMdt[:n, n:]   # upper-right block
    D = eMdt[n:, n:]   # exp(-A_bi^T * dt),  D^{-1} = F_bi^T
    A_d = B
    Q_d = C @ np.linalg.inv(D)   # = C @ F_bi^T
    Q_d = 0.5 * (Q_d + Q_d.T)   # enforce symmetry
    return A_d, Q_d


class SSM_Matern(ProblemInterface, SINS):
    """
    Error-state Kalman filter (Sola-style) with Matern 3/2 GP prior on IMU bias.

    Error state vector (21-D) — matches theory ordering:
        POS     [0:3]   position error
        VEL     [3:6]   velocity error
        ATT     [6:9]   attitude error angles
        BAA_DOT [9:12]  accelerometer bias rate  delta_b_a_dot  (Matern augmented)
        BAA     [12:15] accelerometer bias        delta_b_a
        BGA_DOT [15:18] gyroscope bias rate       delta_b_omega_dot
        BGA     [18:21] gyroscope bias            delta_b_omega

    Theory reference (nominal state):
        x = [p, v, q, b_a, b_a_dot, b_omega, b_omega_dot]^T
    Theory reference (error state):
        delta_x = [delta_p, delta_v, delta_theta,
                   delta_b_a_dot, delta_b_a,
                   delta_b_omega_dot, delta_b_omega]^T

    Non-zero blocks of F_delta_x (theory):
        F[delta_v, delta_theta]   = -C_nb [a_hat]_x dt
        F[delta_v, delta_b_a]     = -C_nb dt
        F[delta_theta, delta_theta] = R^T_{omega_hat dt}
        F[delta_theta, delta_b_omega] = -I dt
        F[delta_b_a_dot/b_a block, same block] = exp(A_bi dt)  (reordered)
    """

    def __init__(self, Ts: float, imu_uncal: IMU,
                 init_pos=xp.array([[0.0], [0.0], [0.0]]),
                 init_yaw: float = 0.0,
                 still: int = 20,
                 **matern_kwargs):
        SINS.__init__(self, Ts)

        self.n_frame = 'ENU'
        self.gravity = xp.array([[0.0], [0.0], [-self.g]])

        # ---- Error-state indices (match theory ordering) ----
        self.POS     = slice(0,  3)
        self.VEL     = slice(3,  6)
        self.ATT     = slice(6,  9)
        self.BAA_DOT = slice(9,  12)  # delta_b_a_dot  (rate first, per theory)
        self.BAA     = slice(12, 15)  # delta_b_a
        self.BGA_DOT = slice(15, 18)  # delta_b_omega_dot
        self.BGA     = slice(18, 21)  # delta_b_omega

        self.INER_DIM = 21
        self.stateDim = 21

        self.STATE_PART_NAMES = ["POS", "VEL", "ATT", "BAA_DOT", "BAA", "BGA_DOT", "BGA"]
        self.STATE_PART_SIZES = [3, 3, 3, 3, 3, 3, 3]

        # ---- Process noise index structure ----
        self.Q_ACC   = slice(0,  3)
        self.Q_GYRO  = slice(3,  6)
        self.Q_BAA_1 = slice(6,  9)   # Matern acc bias noise channel 1
        self.Q_BAA_2 = slice(9,  12)  # Matern acc bias noise channel 2
        self.Q_BGA_1 = slice(12, 15)  # Matern gyro bias noise channel 1
        self.Q_BGA_2 = slice(15, 18)  # Matern gyro bias noise channel 2
        self.Q_DIM   = 18

        # ---- Initial covariance settings ----
        self.noiseInitialPos = 1e-5
        self.noiseInitialVel = 1e-5
        self.noiseInitialOri = xp.pi / 180 * 0.1
        self.noiseInitialBGA = 0.3 * xp.pi / 180   # after static calibration, gyro bias well known
        self.noiseInitialBAA = 0.3                  # after static calibration, acc bias reasonably known

        # ---- IMU-level process noise (same as SSM_Sola) ----
        self.noiseProcessAcc  = 100.0
        self.noiseProcessGyro = 50.0

        # ---- Matern 3/2 GP hyperparameters ----
        # Slow-varying, tight-prior bias: nearly-constant bias with small uncertainty.
        # Large lengthscale → A_d ≈ I → bias changes very slowly between steps.
        # Small sigma → tight stationary prior → filter resists bias drift.
        self.matern_lengthscale_acc  = 50.0           # [s]
        self.matern_lengthscale_gyro = 50.0           # [s]
        self.matern_sigma_acc  = 0.05                 # [m/s^2]  typical IMU bias magnitude
        self.matern_sigma_gyro = 0.1 * np.pi / 180   # [rad/s]

        # ---- Apply any overrides from caller ----
        _allowed = {
            'matern_lengthscale_acc', 'matern_lengthscale_gyro',
            'matern_sigma_acc', 'matern_sigma_gyro',
            'noiseInitialBAA', 'noiseInitialBGA',
            'noiseInitialPos', 'noiseInitialVel', 'noiseInitialOri',
            'noiseProcessAcc', 'noiseProcessGyro',
        }
        for k, v in matern_kwargs.items():
            if k not in _allowed:
                raise KeyError(f"Unknown SSM_Matern parameter '{k}'")
            setattr(self, k, v)

        # ---- State and covariance initialisation ----
        self.m    = xp.zeros((self.stateDim, 1))
        self.qua0 = xp.array([[1.0], [0.0], [0.0], [0.0]])
        self.m[self.POS] = xp.copy(init_pos)

        self.set_state_covariance()
        self.set_fixed_process_noise_covariance()

        # ---- Calibrate IMU ----
        imu, bG = calibrate_my_imu(imu_uncal, still=still, g=self.g)
        self.u_all = xp.vstack((imu.acc, imu.gyr))
        print(f"IMU calibrated. Gyro bias: {bG.reshape(3,) / np.pi * 180} deg/s")

        # ---- Initial alignment ----
        acc_for_init = imu.acc[:, :still]
        self.qua0, self.att0 = self.init_attitude_from_acc(acc_for_init, init_yaw)
        self.m[self.ATT] = xp.copy(self.att0)
        self.quat = xp.copy(self.qua0)

        self.m0 = xp.copy(self.m)
        self.P0 = xp.copy(self.P)
        self.Q0 = xp.copy(self.Q)

        # ---- Build constant matrices ----
        with xp.no_grad():
            # Matern 3/2 discrete matrices in natural [b; b_dot] ordering
            Ad_acc,  Qd_acc  = _matern32_discrete(
                self.matern_lengthscale_acc,
                float(self.matern_sigma_acc)**2,
                Ts)
            Ad_gyro, Qd_gyro = _matern32_discrete(
                self.matern_lengthscale_gyro,
                float(self.matern_sigma_gyro)**2,
                Ts)

            # Store A_d elements (in [b; b_dot] ordering)
            self._a11_acc  = float(Ad_acc[0, 0])
            self._a12_acc  = float(Ad_acc[0, 1])
            self._a21_acc  = float(Ad_acc[1, 0])
            self._a22_acc  = float(Ad_acc[1, 1])

            self._a11_gyro = float(Ad_gyro[0, 0])
            self._a12_gyro = float(Ad_gyro[0, 1])
            self._a21_gyro = float(Ad_gyro[1, 0])
            self._a22_gyro = float(Ad_gyro[1, 1])

            # Permute Q_d to theory's error-state ordering [b_dot; b]
            # Q_d is in [b; b_dot] order from _matern32_discrete
            # Q_d_reordered = P @ Q_d @ P^T  where P = [[0,1],[1,0]]
            P_perm = np.array([[0., 1.], [1., 0.]])
            Qd_acc_reordered  = P_perm @ Qd_acc  @ P_perm.T
            Qd_gyro_reordered = P_perm @ Qd_gyro @ P_perm.T

            # Cholesky in theory's [b_dot; b] ordering
            L_acc  = np.linalg.cholesky(Qd_acc_reordered)
            L_gyro = np.linalg.cholesky(Qd_gyro_reordered)

            # ---- Fx_base: constant parts of the error-state transition ----
            self.Fx_base = xp.eye(self.stateDim)
            self.Fx_base[self.POS, self.VEL] = self.I3 * Ts
            # F[delta_theta, delta_b_omega] = -I*dt  (theory eq.)
            self.Fx_base[self.ATT, self.BGA] = -self.I3 * Ts

            # Matern 3/2 bias transition in [b_dot; b] error-state ordering.
            # In error state [b_dot_k; b_k], the SDE gives:
            #   b_dot_{k+1} = a22*b_dot_k + a21*b_k
            #   b_{k+1}     = a12*b_dot_k + a11*b_k
            # So A_d_reordered = [[a22, a21], [a12, a11]]
            self.Fx_base[self.BAA_DOT, self.BAA_DOT] = self.I3 * self._a22_acc
            self.Fx_base[self.BAA_DOT, self.BAA]     = self.I3 * self._a21_acc
            self.Fx_base[self.BAA,     self.BAA_DOT] = self.I3 * self._a12_acc
            self.Fx_base[self.BAA,     self.BAA]     = self.I3 * self._a11_acc

            self.Fx_base[self.BGA_DOT, self.BGA_DOT] = self.I3 * self._a22_gyro
            self.Fx_base[self.BGA_DOT, self.BGA]     = self.I3 * self._a21_gyro
            self.Fx_base[self.BGA,     self.BGA_DOT] = self.I3 * self._a12_gyro
            self.Fx_base[self.BGA,     self.BGA]     = self.I3 * self._a11_gyro

            # ---- Fi_base: noise input Jacobian (Cholesky in [b_dot; b] ordering) ----
            # F[delta_v, delta_na] = I,  F[delta_theta, n_omega] = I  (theory)
            # F[delta_b_dot_i, delta_n_bi] = I  ->  factored via Cholesky of Q_d_reordered
            self.Fi_base = xp.zeros((self.stateDim, self.Q_DIM))
            self.Fi_base[self.VEL, self.Q_ACC]  = self.I3
            self.Fi_base[self.ATT, self.Q_GYRO] = self.I3

            # Acc bias noise (in [b_dot; b] ordering):
            #   row b_dot (BAA_DOT): L[0,0]*w1
            #   row b     (BAA):     L[1,0]*w1 + L[1,1]*w2
            self.Fi_base[self.BAA_DOT, self.Q_BAA_1] = self.I3 * float(L_acc[0, 0])
            self.Fi_base[self.BAA,     self.Q_BAA_1] = self.I3 * float(L_acc[1, 0])
            self.Fi_base[self.BAA,     self.Q_BAA_2] = self.I3 * float(L_acc[1, 1])

            # Gyro bias noise (in [b_dot; b] ordering)
            self.Fi_base[self.BGA_DOT, self.Q_BGA_1] = self.I3 * float(L_gyro[0, 0])
            self.Fi_base[self.BGA,     self.Q_BGA_1] = self.I3 * float(L_gyro[1, 0])
            self.Fi_base[self.BGA,     self.Q_BGA_2] = self.I3 * float(L_gyro[1, 1])

            # Hx for ZUPT: maps nominal (quaternion) state to velocity measurement
            Hx = xp.zeros((3, self.stateDim + 1))   # +1 for quaternion vs angle dim
            Hx[:, self.VEL] = self.I3
            self._Hx = Hx

    # ------------------------------------------------------------------
    # ProblemInterface required methods
    # ------------------------------------------------------------------

    def set_state_covariance(self, **kwargs):
        if xp._xp.use_torch:
            print("Now using torch")

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        lam_acc  = np.sqrt(3.0) / float(getattr(self, 'matern_lengthscale_acc',  15.0))
        lam_gyro = np.sqrt(3.0) / float(getattr(self, 'matern_lengthscale_gyro', 15.0))
        sig_acc  = float(getattr(self, 'matern_sigma_acc',  0.3))
        sig_gyro = float(getattr(self, 'matern_sigma_gyro', 0.5 * np.pi / 180))

        self.P = xp.zeros((self.stateDim, self.stateDim))
        self.P[self.POS,     self.POS]     = self.I3 * self.noiseInitialPos**2
        self.P[self.VEL,     self.VEL]     = self.I3 * self.noiseInitialVel**2
        self.P[self.ATT,     self.ATT]     = self.I3 * self.noiseInitialOri**2
        self.P[self.BAA,     self.BAA]     = self.I3 * self.noiseInitialBAA**2
        self.P[self.BGA,     self.BGA]     = self.I3 * self.noiseInitialBGA**2
        # Stationary Matern 3/2 variance of the bias rate: sigma^2 * lambda^2
        self.P[self.BAA_DOT, self.BAA_DOT] = self.I3 * (sig_acc  * lam_acc)**2
        self.P[self.BGA_DOT, self.BGA_DOT] = self.I3 * (sig_gyro * lam_gyro)**2

    def set_fixed_process_noise_covariance(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Unknown noise parameter '{key}'")

        dt = self.Ts
        Q  = xp.zeros((self.Q_DIM, self.Q_DIM))

        # IMU measurement noise (same as SSM_Sola: sigma^2 * dt^2)
        Q[self.Q_ACC,   self.Q_ACC]   = self.I3 * self.noiseProcessAcc**2  * dt**2
        Q[self.Q_GYRO,  self.Q_GYRO]  = self.I3 * self.noiseProcessGyro**2 * dt**2
        # Matern bias channels: unit variance; all scaling absorbed into Fi_base (Cholesky)
        Q[self.Q_BAA_1, self.Q_BAA_1] = self.I3
        Q[self.Q_BAA_2, self.Q_BAA_2] = self.I3
        Q[self.Q_BGA_1, self.Q_BGA_1] = self.I3
        Q[self.Q_BGA_2, self.Q_BGA_2] = self.I3

        self.Q = Q

    def transition_model(self):
        dt = self.Ts

        # Bias-corrected IMU measurements
        # BAA = b_a (bias value), BGA = b_omega (bias value)
        a_h     = self.u['acc']  - self.m[self.BAA]
        omega_h = self.u['gyro'] - self.m[self.BGA]

        p = self.m[self.POS, :]
        v = self.m[self.VEL, :]
        q = self.quat

        C = self.qua2dcm(q, 'Cnb')

        # Quaternion integration
        q_incre    = self.rotation_vec2quaternion(omega_h * dt)
        self.quat  = self.quaternion_product(q, q_incre)

        # Nominal state propagation
        m_new = xp.copy(self.m)
        m_new[self.POS] = p + v * dt + 0.5 * (C @ a_h + self.gravity) * dt**2
        m_new[self.VEL] = v + (C @ a_h + self.gravity) * dt
        m_new[self.ATT] = self.qua2att(self.quat)

        # Matern 3/2 bias propagation in theory's error-state ordering [b_dot; b].
        # Reading current states: BAA_DOT=b_dot, BAA=b (acc); BGA_DOT=b_dot, BGA=b (gyro)
        # SDE transition [b; b_dot] -> A_d @ [b; b_dot], reordered to [b_dot; b]:
        #   b_dot_{k+1} = a22*b_dot_k + a21*b_k
        #   b_{k+1}     = a12*b_dot_k + a11*b_k
        baa_dot = self.m[self.BAA_DOT, :]   # b_a_dot
        baa     = self.m[self.BAA,     :]   # b_a
        bga_dot = self.m[self.BGA_DOT, :]   # b_omega_dot
        bga     = self.m[self.BGA,     :]   # b_omega

        m_new[self.BAA_DOT] = self._a22_acc  * baa_dot + self._a21_acc  * baa
        m_new[self.BAA]     = self._a12_acc  * baa_dot + self._a11_acc  * baa
        m_new[self.BGA_DOT] = self._a22_gyro * bga_dot + self._a21_gyro * bga
        m_new[self.BGA]     = self._a12_gyro * bga_dot + self._a11_gyro * bga

        self.m = m_new

        # Error-state Jacobian (time-varying parts injected into Fx_base copy)
        R_wdt = self.rotation_vec2rotation_matrix(omega_h * dt)
        S     = self.skew(a_h)

        _Fx = xp.copy(self.Fx_base)
        # F[delta_v, delta_theta] = -C_nb [a_hat]_x dt
        _Fx[self.VEL, self.ATT] = -C @ S * dt
        # F[delta_v, delta_b_a]  = -C_nb dt  (BAA = b_a at indices 12:15)
        _Fx[self.VEL, self.BAA] = -C * dt
        # F[delta_theta, delta_theta] = R^T_{omega_hat dt}
        _Fx[self.ATT, self.ATT] = R_wdt.T

        self.dfdx = _Fx
        self.dfdq = self.Fi_base

    def set_measurement_fixed_part(self, measure_type):
        self.measureType = measure_type
        if measure_type == 'zupt':
            self.R    = self.I3 * self.zuptR
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
        measure_type = self.measureType.lower()

        if measure_type == 'zupt':
            measure_pre = xp.copy(self.m[self.VEL])

            q = self.quat.flatten()
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]

            # Jacobian of quaternion w.r.t. 3-D error angle (Sola eq. 281)
            Qdtheta = 0.5 * xp.array([
                [-qx, -qy, -qz],
                [ qw, -qz,  qy],
                [ qz,  qw, -qx],
                [-qy,  qx,  qw]
            ])

            # Map from 21-D error state to 22-D nominal (quaternion) state
            # block_diag: [I6, Qdtheta(4x3), I12]  =>  (22 x 21)
            Xdx = xp.block_diag(xp.eye(6), Qdtheta, xp.eye(12))
            self.dhdx = self._Hx @ Xdx

        elif measure_type == 'position':
            measure_pre = xp.copy(self.m[self.POS])
        else:
            raise ValueError(f"Unknown measurement type: {measure_type}")

        return measure_pre

    def comp_internal_states(self, dx):
        epsilon = dx[self.ATT]
        delta_q = self.rotation_vec2quaternion(epsilon)

        x_old = xp.copy(self.m).reshape(-1, 1)
        q_old = xp.copy(self.quat).reshape(-1, 1)
        self.quat = self.quaternion_product(q_old, delta_q)

        m_new          = x_old + dx.reshape(-1, 1)
        m_new[self.ATT] = self.qua2att(self.quat)
        self.m = m_new
