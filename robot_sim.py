import os
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rsc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional


class SixRArmKpEstimator:
    def __init__(self, urdf_path: str, tip_link_name: str = "link6"):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.tip_link_name = tip_link_name
        self.joint_names = self.model.names[1:]

    # ---------- equilibrium solver ----------
    def calc_equilibrium_s1(self, q_des, kp_vec, maxiter=100, eps=1e-12):
        k_mat = np.diag(kp_vec)
        cos_q, sin_q = np.cos(q_des.copy()), np.sin(q_des.copy())

        for _ in range(maxiter):
            q = np.arctan2(sin_q, cos_q)
            f_tilde = pin.computeGeneralizedGravity(self.model, self.data, q) + k_mat.dot(q - q_des)
            dgdq = pin.computeGeneralizedGravityDerivatives(self.model, self.data, q)
            dfdq = dgdq + k_mat

            n = self.nv
            d_chart = np.zeros((n, 2 * n))
            denom = cos_q**2 + sin_q**2
            d_chart[np.arange(n), 2 * np.arange(n)] = -sin_q / denom
            d_chart[np.arange(n), 2 * np.arange(n) + 1] = cos_q / denom

            dftilde_dchart = dfdq.dot(d_chart)
            dq_chart = np.linalg.pinv(dftilde_dchart).dot(f_tilde)

            cos_u = cos_q - dq_chart[0::2]
            sin_u = sin_q - dq_chart[1::2]
            nrm = np.sqrt(cos_u**2 + sin_u**2)
            cos_q, sin_q = cos_u / nrm, sin_u / nrm

            if np.linalg.norm(f_tilde) < eps:
                break
        return np.arctan2(sin_q, cos_q)

    # ---------- quaternion ----------
    def link_quaternion_wxyz(self, q, link_name):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self.model.getFrameId(link_name)
        r_m = self.data.oMf[fid].rotation
        q_xyzw = Rsc.from_matrix(r_m).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        return q_wxyz / np.linalg.norm(q_wxyz)

    # ---------- objective ----------
    @staticmethod
    def _objective_log_kp(log_kp, a_mat, qdes_list, instance, link_name, l2_reg=1e-6):
        kp_vec = np.exp(log_kp)
        j_val = 0.0
        for q_des in qdes_list:
            q_eq = instance.calc_equilibrium_s1(q_des, kp_vec)
            q_wxyz = instance.link_quaternion_wxyz(q_eq, link_name)
            j_val += float(q_wxyz.T @ a_mat @ q_wxyz)
        j_val += l2_reg * float(np.dot(kp_vec, kp_vec))
        return j_val

    def estimate_kp(self, a_mat, qdes_list, link_name: Optional[str] = None,
                    kp_init: Optional[np.ndarray] = None, solver="L-BFGS-B"):
        link = link_name or self.tip_link_name
        if kp_init is None:
            kp_init = np.ones(self.nv) * 10.0
        log_kp0 = np.log(np.clip(kp_init, 1e-9, None))
        fun = lambda x: SixRArmKpEstimator._objective_log_kp(x, -a_mat, qdes_list, self, link)
        res = minimize(fun, log_kp0, method=solver)
        return np.exp(res.x), res

    # ---------- geometry ----------
    def joint_positions(self, q, link_name_tip: Optional[str] = None):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        pts = []
        for jid in range(1, self.model.njoints):
            pts.append(self.data.oMi[jid].translation)
        tip_name = link_name_tip or self.tip_link_name
        fid_tip = self.model.getFrameId(tip_name)
        pts.append(self.data.oMf[fid_tip].translation)
        return np.vstack(pts)

    def plot_arm_positions(self, ax, points, style_kwargs=None, label=None):
        if style_kwargs is None:
            style_kwargs = {}
        ax.plot(points[:, 0], points[:, 1], points[:, 2],
                marker="o", **style_kwargs, label=label)

    def draw_frames(self, ax, q, link_name_tip: Optional[str] = None, axis_length=0.05):
        """Draw coordinate frames for each joint and tip link."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        colors = ["r", "g", "b"]  # x,y,z
        # joints
        for jid in range(1, self.model.njoints):
            o_m_i = self.data.oMi[jid]
            origin = o_m_i.translation
            rot = o_m_i.rotation
            for k, c in enumerate(colors):
                axis = rot[:, k] * axis_length
                ax.plot([origin[0], origin[0] + axis[0]],
                        [origin[1], origin[1] + axis[1]],
                        [origin[2], origin[2] + axis[2]], c)

        # tip link
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

    def visualize_compare(self, q_des, q_eq,
                          link_name_tip: Optional[str] = None,
                          title="6R arm: q_des vs q_eq"):
        p_des = self.joint_positions(q_des, link_name_tip)
        p_eq = self.joint_positions(q_eq, link_name_tip)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        self.plot_arm_positions(ax, p_des, {"linestyle": "--"}, label="q_des")
        self.plot_arm_positions(ax, p_eq, {"linewidth": 2}, label="q_eq")

        # draw frames for equilibrium pose
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
        plt.gca().set_box_aspect([1,1,1])
        plt.tight_layout()
        plt.show()


# ----------------- Example -----------------
if __name__ == "__main__":
    urdf_path = os.path.abspath("simple6r.urdf")
    estimator = SixRArmKpEstimator(urdf_path)

    A = np.diag([0.0, -1000.0, -200.0, -100.0])

    rng = np.random.default_rng(0)
    qdes_list = [rng.uniform(low=-1.0, high=1.0, size=estimator.nv) for _ in range(8)]

    kp0 = np.array([20, 15, 15, 8, 6, 4], dtype=float)
    kp_hat, res = estimator.estimate_kp(A, qdes_list, kp_init=kp0)
    print("Kp_hat =", kp_hat)
    print("opt_status =", res.message)

    q_des_demo = qdes_list[0]
    q_eq_demo = estimator.calc_equilibrium_s1(q_des_demo, kp_hat)
    estimator.visualize_compare(q_des_demo, q_eq_demo)
