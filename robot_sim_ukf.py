# Sequential Kp estimation with UKF (scalar measurement): y = q^T A q
# Requirements: pinocchio, numpy, scipy, matplotlib
import os
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rsc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional

# ------------------------ Quaternion helpers for Bingham A ------------------------
class Quaternion:
    def __init__(self, w, x, y, z):
        self._array = np.array([w, x, y, z], dtype=float)
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def array(self):
        return self._array

def Lmat(quat: Quaternion) -> np.ndarray:
    a, b, c, d = quat.array()
    return np.array([
        [a, -b, -c, -d],
        [b,  a, -d,  c],
        [c,  d,  a, -b],
        [d, -c,  b,  a]
    ])

def Rmat(quat: Quaternion) -> np.ndarray:
    w, x, y, z = quat.array()
    return np.array([
        [w, -x, -y, -z],
        [x,  w,  z, -y],
        [y, -z,  w,  x],
        [z,  y, -x,  w]
    ])

def simple_bingham_unit(before_vec3: np.ndarray, after_vec3: np.ndarray, parameter: float = 1000.) -> np.ndarray:
    b = np.asarray(before_vec3, dtype=float)
    a = np.asarray(after_vec3, dtype=float)
    bn = b / (np.linalg.norm(b) + 1e-12)
    an = a / (np.linalg.norm(a) + 1e-12)
    vq = Quaternion(0.0, bn[0], bn[1], bn[2])
    xq = Quaternion(0.0, an[0], an[1], an[2])
    P = Lmat(xq) - Rmat(vq)
    A0 = -0.25 * (P.T @ P)  # negative semidefinite
    return float(parameter) * A0

# ------------------------ SixRArmKpEstimator ------------------------
class SixRArmKpEstimator:
    def __init__(self, urdf_path: str, tip_link_name: str = "link6", base_link_name: str = "base_link"):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.tip_link_name = tip_link_name
        self.base_link_name = base_link_name
        self.joint_names = self.model.names[1:]
        self.base_fid = self.model.getFrameId(self.base_link_name)
        self.tip_fid = self.model.getFrameId(self.tip_link_name)

    # ---------- S^1-Newton equilibrium ----------
    def calc_equilibrium_s1(self, q_des: np.ndarray, kp_vec: np.ndarray, maxiter: int = 100, eps: float = 1e-12) -> np.ndarray:
        k_mat = np.diag(kp_vec)
        cos_q = np.cos(q_des.copy())
        sin_q = np.sin(q_des.copy())

        for _ in range(maxiter):
            q = np.arctan2(sin_q, cos_q)
            f_tilde = pin.computeGeneralizedGravity(self.model, self.data, q) + k_mat.dot(q - q_des)
            dgdq = pin.computeGeneralizedGravityDerivatives(self.model, self.data, q)
            dfdq = dgdq + k_mat

            n = self.nv
            d_chart = np.zeros((n, 2 * n))
            denom = cos_q**2 + sin_q**2
            d_chart[np.arange(n), 2 * np.arange(n)] = -sin_q / denom
            d_chart[np.arange(n), 2 * np.arange(n) + 1] =  cos_q / denom

            dftilde_dchart = dfdq.dot(d_chart)
            dq_chart = np.linalg.pinv(dftilde_dchart).dot(f_tilde)

            cos_u = cos_q - dq_chart[0::2]
            sin_u = sin_q - dq_chart[1::2]
            nrm = np.sqrt(cos_u**2 + sin_u**2)
            cos_q, sin_q = cos_u / nrm, sin_u / nrm

            if np.linalg.norm(f_tilde) < eps:
                break

        return np.arctan2(sin_q, cos_q)

    # ---------- EE rotation / quaternion in base frame ----------
    def ee_rotation_in_base(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        R_wb = self.data.oMf[self.base_fid].rotation
        R_we = self.data.oMf[self.tip_fid].rotation
        return R_wb.T @ R_we  # base -> ee

    def ee_quaternion_wxyz_base(self, q: np.ndarray) -> np.ndarray:
        R_be = self.ee_rotation_in_base(q)
        q_xyzw = Rsc.from_matrix(R_be).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        return q_wxyz / np.linalg.norm(q_wxyz)

    # ---------- gravity direction in EE ----------
    def gravity_dir_in_ee(self, q: np.ndarray, g_base: np.ndarray) -> np.ndarray:
        R_be = self.ee_rotation_in_base(q)
        gb_unit = g_base / (np.linalg.norm(g_base) + 1e-12)
        gee = R_be @ gb_unit
        return gee / (np.linalg.norm(gee) + 1e-12)

    # ---------- dataset: build dynamic A per time-step from true Kp* ----------
    def build_dynamicA_timeseries(self, qdes_seq: list, kp_true: np.ndarray, g_base: np.ndarray, parameter: float = 1000.0):
        A_seq = []
        qeq_true_seq = []
        for q_des in qdes_seq:
            q_eq_true = self.calc_equilibrium_s1(q_des, kp_true)
            gee_true = self.gravity_dir_in_ee(q_eq_true, g_base)
            A_t = simple_bingham_unit(g_base, gee_true, parameter=parameter)
            A_seq.append(A_t)
            qeq_true_seq.append(q_eq_true)
        return A_seq, qeq_true_seq

    # ---------- geometry & visualization ----------
    def joint_positions(self, q: np.ndarray, link_name_tip: Optional[str] = None) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pts = []
        for jid in range(1, self.model.njoints):
            pts.append(self.data.oMi[jid].translation)
        tip_name = link_name_tip or self.tip_link_name
        fid_tip = self.model.getFrameId(tip_name)
        pts.append(self.data.oMf[fid_tip].translation)
        return np.vstack(pts)

    def plot_arm_positions(self, ax, points: np.ndarray, style_kwargs=None, label=None):
        if style_kwargs is None:
            style_kwargs = {}
        ax.plot(points[:, 0], points[:, 1], points[:, 2], marker="o", **style_kwargs, label=label)

    def draw_frames(self, ax, q: np.ndarray, link_name_tip: Optional[str] = None, axis_length=0.05):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        colors = ["r", "g", "b"]  # x, y, z
        for jid in range(1, self.model.njoints):
            o_m_i = self.data.oMi[jid]
            origin = o_m_i.translation
            rot = o_m_i.rotation
            for k, c in enumerate(colors):
                axis = rot[:, k] * axis_length
                ax.plot([origin[0], origin[0] + axis[0]],
                        [origin[1], origin[1] + axis[1]],
                        [origin[2], origin[2] + axis[2]], c)
        tip_name = link_name_tip or self.tip_link_name
        fid_tip = self.model.getFrameId(tip_name)
        o_m_f = self.data.oMf[fid_tip]
        origin = o_m_f.translation
        rot = o_m_f.rotation
        for k, c in enumerate(colors):
            axis = rot[:, k] * axis_length
            ax.plot([origin[0], origin[0] + axis[0]],
                    [origin[1], origin[1] + axis[1]],
                    [origin[2], origin[2] + axis[2]], c)

    def visualize_compare(self, q_des: np.ndarray, q_eq: np.ndarray, link_name_tip: Optional[str] = None, title: str = "6R arm: q_des vs q_eq"):
        p_des = self.joint_positions(q_des, link_name_tip)
        p_eq = self.joint_positions(q_eq, link_name_tip)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        self.plot_arm_positions(ax, p_des, {"linestyle": "--"}, label="q_des")
        self.plot_arm_positions(ax, p_eq, {"linewidth": 2}, label="q_eq")
        self.draw_frames(ax, q_eq, link_name_tip, axis_length=0.05)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(title)
        ax.legend()

        all_pts = np.vstack([p_des, p_eq])
        center = all_pts.mean(axis=0)
        radius = np.max(np.linalg.norm(all_pts - center, axis=1))
        radius = max(radius, 0.2)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=25, azim=45)
        plt.gca().set_box_aspect([1, 1, 1])
        plt.tight_layout()
        plt.show()

# ------------------------ UKF for x = log(Kp) with scalar measurement ------------------------
class LogKpUKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R_scalar: float,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R = float(R_scalar)
        self.n = x0.size
        # UT params
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = alpha**2 * (self.n + kappa) - self.n
        self.c = self.n + self.lmbda
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2.0 * self.c))
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2.0 * self.c))
        self.Wm[0] = self.lmbda / self.c
        self.Wc[0] = self.lmbda / self.c + (1.0 - alpha**2 + beta)

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        n = self.n
        jitter = 1e-10
        # ensure PSD
        Psym = 0.5 * (P + P.T) + jitter * np.eye(n)
        try:
            S = np.linalg.cholesky(self.c * Psym)
        except np.linalg.LinAlgError:
            # add more jitter if needed
            S = np.linalg.cholesky(self.c * (Psym + 1e-8 * np.eye(n)))
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            col = S[:, i]
            sigmas[1 + i] = x + col
            sigmas[1 + n + i] = x - col
        return sigmas

    def predict(self):
        # random walk: x = x, P = P + Q
        self.P = self.P + self.Q

    def update(self, z_t: float, q_des_t: np.ndarray, A_t: np.ndarray, estimator: SixRArmKpEstimator):
        # sigma points
        sigmas = self._sigma_points(self.x, self.P)

        # propagate through measurement: h(x) = q(x)^T A_t q(x)
        y_sig = np.zeros(sigmas.shape[0])
        for i, xs in enumerate(sigmas):
            kp_vec = np.exp(xs)
            q_eq = estimator.calc_equilibrium_s1(q_des_t, kp_vec)
            q_wxyz = estimator.ee_quaternion_wxyz_base(q_eq)
            y_sig[i] = float(q_wxyz.T @ A_t @ q_wxyz)

        # predicted measurement
        y_pred = np.dot(self.Wm, y_sig)

        # innovation covariance S (scalar) and cross-covariance C (n x 1)
        dy = y_sig - y_pred
        S = float(np.dot(self.Wc, dy * dy)) + self.R
        C = np.sum(self.Wc[:, None] * (sigmas - self.x) * dy[:, None], axis=0).reshape(self.n, 1)

        # Kalman gain (n x 1)
        K = C / (S + 1e-18)

        # update state
        self.x = self.x + (K.flatten() * (z_t - y_pred))
        # update covariance
        self.P = self.P - K @ K.T * S
        # symmetrize
        self.P = 0.5 * (self.P + self.P.T)

# ------------------------ Example usage: sequential UKF ------------------------
if __name__ == "__main__":
    # Assume simple6r.urdf exists
    urdf_path = os.path.abspath("simple6r.urdf")
    est = SixRArmKpEstimator(urdf_path, tip_link_name="link6", base_link_name="base_link")

    # ground-truth Kp*
    kp_true = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float)

    # gravity in base
    g_base = np.array([0.0, 0.0, -9.8])

    # build a slowly varying time series of q_des (quasi-static)
    T = 60
    rng = np.random.default_rng(2)
    q_des = rng.uniform(-0.5, 0.5, size=est.nv)
    qdes_seq = []
    for t in range(T):
        dq = 0.05 * rng.standard_normal(est.nv)  # small step
        q_des = np.clip(q_des + dq, -1.2, 1.2)
        qdes_seq.append(q_des.copy())

    # construct A_t from true Kp* (measurement-origin, kept fixed per t)
    A_seq, qeq_true_seq = est.build_dynamicA_timeseries(qdes_seq, kp_true, g_base, parameter=1000.0)

    # UKF init on x = log(Kp): start "large-ish" but stable
    x0 = np.log(np.ones(est.nv) * 25.0)   # e.g., center 25
    P0 = np.eye(est.nv) * 1.0             # initial covariance
    Q = np.eye(est.nv) * 1e-3             # process noise (state random-walk)
    R_scalar = 1e-6                        # measurement noise variance (scalar)

    ukf = LogKpUKF(x0, P0, Q, R_scalar, alpha=1e-2, beta=2.0, kappa=0.0)

    # run filter
    kph_seq = []
    for t in range(T):
        ukf.predict()
        # target scalar measurement is y=0 (since A_t built from "true" gravity mapping)
        z_t = 0.0
        ukf.update(z_t, qdes_seq[t], A_seq[t], est)
        kph_seq.append(np.exp(ukf.x).copy())

    kp_hat = np.exp(ukf.x)
    print("Kp_true =", kp_true)
    print("Kp_hat  =", kp_hat)
    rel_err = np.linalg.norm(kp_hat - kp_true) / (np.linalg.norm(kp_true) + 1e-12)
    print("relative_error =", rel_err)

    # visualize one step (final)
    q_eq_demo = est.calc_equilibrium_s1(qdes_seq[-1], kp_hat)
    est.visualize_compare(qdes_seq[-1], q_eq_demo, title="Equilibrium with UKF-estimated Kp")
