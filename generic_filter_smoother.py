import xp
from abc import ABC, abstractmethod


class ProblemInterface(ABC):
    @abstractmethod
    def set_state_covariance(self):
        pass

    @abstractmethod
    def set_fixed_process_noise_covariance(self):
        pass

    @abstractmethod
    def transition_model(self):
        pass

    @abstractmethod
    def measurement_model(self):
        pass


class GenericFilterSmoother:
    """Generic filter and offline RTS smoother for EKF/ErKF."""

    def __init__(self, estimator_type):
        self.estimator_type = estimator_type
        self.S = None
        self.K = None
        self.innov = None
        self.nll_sum = 0.0

    def smoother(self, problem, results):
        data_length = results["m"].shape[1]

        results["ms"] = xp.zeros(results["m"].shape)
        results["ms"][:, -1] = results["m"][:, -1]
        results["Ps"] = xp.zeros(results["COV"].shape)
        results["Ps"][:, :, -1] = results["COV"][:, :, -1]

        problem.ms = xp.copy(results["m"][:, -1]).reshape(-1, 1)
        problem.Ps = xp.copy(results["COV"][:, :, -1])

        if self.estimator_type in ["ErKF", "ErRTS"]:
            problem.delta_ms = xp.zeros(problem.m0.shape)
            results["delta_ms"] = xp.zeros(results["m"].shape)

        for k in range(data_length - 2, -1, -1):
            problem.u = {
                "acc": xp.reshape(problem.u_all[0:3, k], (3, 1)),
                "gyro": xp.reshape(problem.u_all[3:6, k], (3, 1)),
            }

            problem.m = xp.copy(results["m"][:, k]).reshape(-1, 1)
            problem.quat = xp.copy(results["QUA"][:, k]).reshape(-1, 1)
            problem.P = xp.copy(results["COV"][:, :, k])

            self.predict(problem)

            mp = xp.copy(problem.m).reshape(-1, 1)
            quatp = xp.copy(problem.quat).reshape(-1, 1)
            Pp = xp.copy(problem.P)

            Pk = results["COV"][:, :, k]
            F = results["Fx"][:, :, k]

            A = Pk @ F.T
            L = xp.cholesky(Pp)
            Y = xp.solve(L, A.T)
            Ks = xp.solve(L.T, Y).T

            problem.Ps = results["COV"][:, :, k] + Ks @ (problem.Ps - Pp) @ Ks.T

            if self.estimator_type in ["EKF", "ERTS"]:
                problem.ms = results["m"][:, k].reshape(-1, 1) + Ks @ (problem.ms - mp)
            elif self.estimator_type in ["ErKF", "ErRTS"]:
                problem.delta_ms = (
                    results["delta_m"][:, k].reshape(-1, 1)
                    + Ks @ (problem.delta_ms - xp.zeros_like(problem.m0))
                )
                problem.m = xp.copy(mp).reshape(-1, 1)
                problem.quat = xp.copy(quatp).reshape(-1, 1)
                problem.comp_internal_states(problem.delta_ms)
                problem.ms = xp.copy(problem.m).reshape(-1, 1)
                results["delta_ms"][:, k] = xp.copy(problem.delta_ms).flatten()

            results["ms"][:, k] = xp.copy(problem.ms).flatten()
            results["Ps"][:, :, k] = xp.copy(problem.Ps)

        return results

    def predict(self, problem):
        P = xp.copy(problem.P)
        problem.transition_model()
        problem.P = problem.dfdx @ P @ problem.dfdx.T + problem.dfdq @ problem.Q @ problem.dfdq.T
        problem.delta_m = xp.zeros(problem.m0.shape)

    def update(self, problem):
        h = problem.measurement_model()

        m = xp.copy(problem.m)
        P = xp.copy(problem.P)
        H = xp.copy(problem.dhdx)
        M = xp.copy(problem.dhdr)

        self.S = H @ P @ H.T + M @ problem.R @ M.T
        self.S = 0.5 * (self.S + self.S.T) + 1e-9 * xp.eye(self.S.shape[0])

        L = xp.cholesky(self.S)
        self.K = xp.solve(L.T, xp.solve(L, H @ P.T)).T
        self.innov = xp.array(problem.measurement.reshape(-1, 1)) - h

        problem.P = P - self.K @ self.S @ self.K.T

        if self.estimator_type == "EKF":
            problem.m = m + self.K @ self.innov
        elif self.estimator_type == "ErKF":
            problem.delta_m = self.K @ self.innov
            problem.comp_internal_states(problem.delta_m)
            problem.P = problem.error_state_reset(problem.delta_m, problem.P)

        problem.P = self.maintain_positive_semidefinite(problem.P)

    def maintain_positive_semidefinite(self, P):
        return 0.5 * (P + P.T)

    def nll_k(self):
        n = self.S.shape[0]
        _, logdetS = xp.slogdet(self.S)
        logdet = logdetS + n * xp.log(2 * xp.pi)
        L = xp.cholesky(self.S)
        mahalanobis = self.innov.T @ xp.solve(L.T, xp.solve(L, self.innov))
        return 0.5 * (logdet + mahalanobis).flatten()
