# Kp estimation using S^1-Newton equilibrium + dynamic A from gravity direction
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
    """Return A = parameter * A0, where A0 = -0.25 * P^T P (negative semidefinite).
       before_vec3, after_vec3 are 3D vectors; they will be normalized inside."""
    b = np.asarray(before_vec3, dtype=float)
    a = np.asarray(after_vec3, dtype=float)
    bn = b / (np.linalg.norm(b) + 1e-12)
    an = a / (np.linalg.norm(a) + 1e-12)
    vq = Quaternion(0.0, bn[0], bn[1], bn[2])
    xq = Quaternion(0.0, an[0], an[1], an[2])
    P = Lmat(xq) - Rmat(vq)
    A0 = -0.25 * (P.T @ P)  # max eig of P^T P is 4.0 when inputs are unit
    return float(parameter) * A0


# ------------------------ Estimator class ------------------------
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
        # cache frame ids
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

    # ---------- EE rotation / quaternion (base frame) ----------
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

    # ---------- dataset: build A per sample from true Kp* ----------
    def build_dynamicA_dataset(self, qdes_list: list, kp_true: np.ndarray, g_base: np.ndarray, parameter: float = 1000.0):
        A_list = []
        qeq_true_list = []
        qw_true_list = []
        gee_true_list = []
        for q_des in qdes_list:
            q_eq_true = self.calc_equilibrium_s1(q_des, kp_true)
            gee_true = self.gravity_dir_in_ee(q_eq_true, g_base)
            A = simple_bingham_unit(g_base, gee_true, parameter=parameter)
            A_list.append(A)
            qeq_true_list.append(q_eq_true)
            qw_true_list.append(self.ee_quaternion_wxyz_base(q_eq_true))
            gee_true_list.append(gee_true)
        return A_list, qeq_true_list, qw_true_list, gee_true_list

    # ---------- objective with dynamic A (fixed per-sample) ----------
    @staticmethod
    def _objective_log_kp_dynamic(log_kp: np.ndarray, A_list: list, qdes_list: list, instance: "SixRArmKpEstimator", l2_reg: float = 1e-6) -> float:
        kp_vec = np.exp(log_kp)
        J = 0.0
        for A, q_des in zip(A_list, qdes_list):
            q_eq_hat = instance.calc_equilibrium_s1(q_des, kp_vec)
            q_wxyz_hat = instance.ee_quaternion_wxyz_base(q_eq_hat)
            J += float(q_wxyz_hat.T @ (-A) @ q_wxyz_hat)  # minimize -A == maximize A
        J += l2_reg * float(np.dot(kp_vec, kp_vec))
        return J

    def estimate_kp_dynamicA(self, A_list: list, qdes_list: list, kp_init: Optional[np.ndarray] = None, solver: str = "L-BFGS-B"):
        if kp_init is None:
            kp_init = np.ones(self.nv) * 10.0
        log_kp0 = np.log(np.clip(kp_init, 1e-9, None))
        fun = lambda x: SixRArmKpEstimator._objective_log_kp_dynamic(x, A_list, qdes_list, self)
        res = minimize(fun, log_kp0, method=solver)
        kp_hat = np.exp(res.x)
        return kp_hat, res

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


# ------------------------ Example usage ------------------------
if __name__ == "__main__":
    # Assume simple6r.urdf exists
    urdf_path = os.path.abspath("simple6r.urdf")
    est = SixRArmKpEstimator(urdf_path, tip_link_name="link6", base_link_name="base_link")

    # Hidden ground-truth Kp (to be recovered)
    kp_true = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float)

    # Build sample q_des list
    rng = np.random.default_rng(1)
    qdes_list = [rng.uniform(low=-1.0, high=1.0, size=est.nv) for _ in range(10)]

    # Gravity in base frame (direction only matters; function normalizes)
    g_base = np.array([0.0, 0.0, -9.8])

    # Create per-sample A from gravity mapping under true Kp
    A_list, qeq_true_list, qw_true_list, gee_true_list = est.build_dynamicA_dataset(
        qdes_list, kp_true, g_base, parameter=1000.0
    )

    # Initial guess for Kp
    kp0 = np.array([20, 20, 20, 20, 20, 20], dtype=float)

    # Estimate Kp using dynamic (but fixed-per-sample) A
    kp_hat, res = est.estimate_kp_dynamicA(A_list, qdes_list, kp_init=kp0)
    print("Kp_true =", kp_true)
    print("Kp_hat  =", kp_hat)
    print("opt_status =", res.message)

    # Visualize one sample (compare desired vs. equilibrium under estimated Kp)
    q_des_demo = qdes_list[0]
    q_eq_demo = est.calc_equilibrium_s1(q_des_demo, kp_hat)
    est.visualize_compare(q_des_demo, q_eq_demo, title="Equilibrium with estimated Kp")

    # (Optional) simple accuracy report
    rel_err = np.linalg.norm(kp_hat - kp_true) / (np.linalg.norm(kp_true) + 1e-12)
    print("relative_error =", rel_err)
