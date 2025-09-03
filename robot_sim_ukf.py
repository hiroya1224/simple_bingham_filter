# Sequential Kp estimation with UKF + alternating control:
# (1) rigid IK step -> (2) UKF update -> (3) gravity-aware correction step
#
# Requirements: pinocchio, numpy, scipy, matplotlib
#   pip install pin pinocchio numpy scipy matplotlib
#
# Notes:
# - Generates a minimal 6R URDF locally if missing.
# - Uses S^1-Newton to solve gravity+Kp equilibrium inside the gravity-aware step.
# - Kp from UKF is exponentially smoothed (EMA) only; no extra stabilizers.

import os
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional

# ------------------------ URDF helper ------------------------
def ensure_urdf():
    urdf_path = os.path.abspath("simple6r.urdf")
    if not os.path.exists(urdf_path):
        txt = """<?xml version="1.0"?>
<robot name="simple6r">
  <link name="base_link"/>
  {links}
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.10" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1416" upper="3.1416" effort="150" velocity="2.5"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.20" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.61799" upper="2.61799" effort="120" velocity="2.5"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/><child link="link3"/>
    <origin xyz="0 0 0.20" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.61799" upper="2.61799" effort="120" velocity="2.5"/>
  </joint>
  <joint name="joint4" type="revolute">
    <parent link="link3"/><child link="link4"/>
    <origin xyz="0 0 0.20" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-3.1416" upper="3.1416" effort="60" velocity="3.0"/>
  </joint>
  <joint name="joint5" type="revolute">
    <parent link="link4"/><child link="link5"/>
    <origin xyz="0 0 0.18" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.1416" upper="3.1416" effort="60" velocity="3.0"/>
  </joint>
  <joint name="joint6" type="revolute">
    <parent link="link5"/><child link="link6"/>
    <origin xyz="0 0 0.12" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-3.1416" upper="3.1416" effort="40" velocity="3.5"/>
  </joint>
</robot>""".format(
    links="\n".join([
        f"""
  <link name="link{i}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{1.0 if i<=3 else 0.5}"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="{0.20 if i<=4 else 0.12}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="{0.20 if i<=4 else 0.12}"/></geometry>
    </collision>
  </link>
        """.strip()
        for i in range(1, 7)
    ]))
        with open(urdf_path, "w") as f:
            f.write(txt)
    return urdf_path

def rpy_to_matrix(r, p, y):
    if hasattr(pin, "rpy") and hasattr(pin.rpy, "rpyToMatrix"):
        return pin.rpy.rpyToMatrix(r, p, y)
    else:
        return pin.utils.rpyToMatrix(r, p, y)

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

    def se3_error(self, T_cur: pin.SE3, T_des: pin.SE3) -> np.ndarray:
        E = T_cur.inverse() * T_des
        dp = E.translation
        dth = pin.log3(E.rotation)
        return np.hstack([dp, dth])

    def frame_jacobian_local(self, q: np.ndarray, frame_id: int) -> np.ndarray:
        # ensure joint Jacobians and frame placements are up to date
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        ref = pin.ReferenceFrame.LOCAL if hasattr(pin, "ReferenceFrame") else pin.LOCAL
        J6 = pin.getFrameJacobian(self.model, self.data, frame_id, ref)
        return J6  # (6, nv)

    def draw_pose_axes(self, ax, T: pin.SE3, axis_length: float = 0.07):
        colors = ["r", "g", "b"]
        o = T.translation
        Rm = T.rotation
        for k, c in enumerate(colors):
            a = Rm[:, k] * axis_length
            ax.plot([o[0], o[0]+a[0]], [o[1], o[1]+a[1]], [o[2], o[2]+a[2]], c, linewidth=2.0)
        ax.scatter([o[0]], [o[1]], [o[2]], s=30, c="k")

    def visualize_with_target(self, q_des: np.ndarray, q_eq: np.ndarray, T_des: pin.SE3,
                              link_name_tip: Optional[str] = None,
                              title: str = "Gravity IK (q_des vs q_eq vs target)"):
        p_des = self.joint_positions(q_des, link_name_tip)
        p_eq = self.joint_positions(q_eq, link_name_tip)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        self.plot_arm_positions(ax, p_des, {"linestyle": "--"}, label="q_des")
        self.plot_arm_positions(ax, p_eq, {"linewidth": 2}, label="q_eq")
        self.draw_frames(ax, q_eq, link_name_tip, axis_length=0.05)
        self.draw_pose_axes(ax, T_des, axis_length=0.07)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(title)
        ax.legend()

        all_pts = np.vstack([p_des, p_eq, T_des.translation.reshape(1,3)])
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

    # ---------- Rigid IK (no gravity) ----------
    def ik_step_rigid(self,
        q: np.ndarray,
        T_des: pin.SE3,
        w_pos: float = 1.0,
        w_rot: float = 0.3,
        lambda_damp: float = 1e-4,
        step_scale: float = 1.0,
        use_limits: bool = True,
    ):
        fid = self.tip_fid
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T_cur = self.data.oMf[fid]
        e = self.se3_error(T_cur, T_des)  # (6,)
        J6 = self.frame_jacobian_local(q, fid)  # (6, nv)
        W = np.diag([w_pos, w_pos, w_pos, w_rot, w_rot, w_rot])
        H = J6.T @ W @ J6 + lambda_damp * np.eye(self.nv)
        g = J6.T @ W @ e
        dq = - np.linalg.solve(H, g)
        dq = step_scale * dq
        q_new = q + dq
        if use_limits and hasattr(self.model, "lowerPositionLimit"):
            lo = self.model.lowerPositionLimit; hi = self.model.upperPositionLimit
            q_new = np.minimum(np.maximum(q_new, lo), hi)
        info = {"err_norm": float(np.linalg.norm(e)), "dq_norm": float(np.linalg.norm(dq))}
        return q_new, info

    # ---------- Gravity-aware outer step (simple "lively" version) ----------
    def ik_step_gravity(self,
        q_des: np.ndarray,
        kp_vec: np.ndarray,
        T_des: pin.SE3,
        w_pos: float = 1.0,
        w_rot: float = 0.3,
        lambda_damp: float = 1e-5,
        step_scale: float = 1.0,
        use_limits: bool = True,
    ):
        nv = self.nv
        K = np.diag(kp_vec)
        fid = self.tip_fid

        # inner equilibrium at current q_des
        q_eq = self.calc_equilibrium_s1(q_des, kp_vec)

        # task error at q_eq
        pin.forwardKinematics(self.model, self.data, q_eq)
        pin.updateFramePlacements(self.model, self.data)
        T_cur = self.data.oMf[fid]
        e = self.se3_error(T_cur, T_des)  # (6,)

        # M = J (G+K)^{-1} K
        J6 = self.frame_jacobian_local(q_eq, fid)  # (6, nv)
        G = pin.computeGeneralizedGravityDerivatives(self.model, self.data, q_eq)  # (nv, nv)
        A = G + K
        B = np.linalg.solve(A, K)        # (nv, nv)
        M = J6 @ B                        # (6, nv)

        # damped least squares on q_des
        W = np.diag([w_pos, w_pos, w_pos, w_rot, w_rot, w_rot])
        H = M.T @ W @ M + lambda_damp * np.eye(nv)
        g = M.T @ W @ e
        dq_des = -np.linalg.solve(H, g)

        # update
        dq_des = step_scale * dq_des
        q_des_new = q_des + dq_des

        # joint limits
        if use_limits and hasattr(self.model, "lowerPositionLimit"):
            lo = self.model.lowerPositionLimit
            hi = self.model.upperPositionLimit
            q_des_new = np.minimum(np.maximum(q_des_new, lo), hi)

        info = {
            "err_norm": float(np.linalg.norm(e)),
            "dq_norm": float(np.linalg.norm(dq_des)),
        }
        return q_des_new, q_eq, info

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
        Psym = 0.5 * (P + P.T) + jitter * np.eye(n)
        try:
            S = np.linalg.cholesky(self.c * Psym)
        except np.linalg.LinAlgError:
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
        sigmas = self._sigma_points(self.x, self.P)
        y_sig = np.zeros(sigmas.shape[0])
        for i, xs in enumerate(sigmas):
            kp_vec = np.exp(xs)
            q_eq = estimator.calc_equilibrium_s1(q_des_t, kp_vec)
            q_wxyz = estimator.ee_quaternion_wxyz_base(q_eq)
            y_sig[i] = float(q_wxyz.T @ A_t @ q_wxyz)
        y_pred = np.dot(self.Wm, y_sig)
        dy = y_sig - y_pred
        S = float(np.dot(self.Wc, dy * dy)) + self.R
        C = np.sum(self.Wc[:, None] * (sigmas - self.x) * dy[:, None], axis=0).reshape(self.n, 1)
        K = C / (S + 1e-18)
        self.x = self.x + (K.flatten() * (z_t - y_pred))
        self.P = self.P - K @ K.T * S
        self.P = 0.5 * (self.P + self.P.T)

# ------------------------ Main: alternating control loop ------------------------
if __name__ == "__main__":
    urdf_path = ensure_urdf()
    est = SixRArmKpEstimator(urdf_path, tip_link_name="link6", base_link_name="base_link")

    # ground-truth plant stiffness (unknown to estimator)
    kp_true = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float)
    g_base = np.array([0.0, 0.0, -9.8])

    # target pose
    p_des = np.array([0.35, 0.05, 0.45])
    R_des = rpy_to_matrix(0.0, 0.0, 0.0)
    T_des = pin.SE3(R_des, p_des)

    # UKF on x=log(Kp)
    x0 = np.log(np.ones(est.nv) * 25.0)
    P0 = np.eye(est.nv) * 1.0
    Q = np.eye(est.nv) * 1e-3
    R_scalar = 1e-6
    ukf = LogKpUKF(x0, P0, Q, R_scalar, alpha=1e-2, beta=2.0, kappa=0.0)

    # Kp EMA (smoothing only)
    ema_eta = 0.2
    kp_hat_prev = np.exp(ukf.x).copy()
    kp_min = 1e-6

    # controller reference
    q_des_ctrl = np.zeros(est.nv)

    # loop parameters
    T = 80
    w_pos_rigid, w_rot_rigid = 1.0, 0.3
    w_pos_grav,  w_rot_grav  = 1.0, 0.3
    lam_rigid = 1e-4
    lam_grav  = 1e-5
    step_scale_rigid = 1.0
    step_scale_grav  = 1.0

    # logs
    kph_seq = []
    qeq_seq = []
    err_rig_seq = []
    err_grav_seq = []
    qdes_seq_ctrl = []

    # live viz
    plt.ion()
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")

    for t in range(T):
        # (A) rigid IK step (no gravity)
        q_des_ctrl, info_rig = est.ik_step_rigid(
            q=q_des_ctrl, T_des=T_des,
            w_pos=w_pos_rigid, w_rot=w_rot_rigid,
            lambda_damp=lam_rigid,
            step_scale=step_scale_rigid,
            use_limits=True
        )
        err_rig_seq.append(info_rig["err_norm"])

        # (B) UKF update from "plant" scalar measurement at current reference
        q_eq_true = est.calc_equilibrium_s1(q_des_ctrl, kp_true)
        gee_true = est.gravity_dir_in_ee(q_eq_true, g_base)
        A_t = simple_bingham_unit(g_base, gee_true, parameter=1000.0)
        ukf.predict()
        z_t = 0.0
        ukf.update(z_t, q_des_ctrl, A_t, est)

        # EMA smoothing of Kp
        kp_raw = np.exp(ukf.x)
        kp_hat_t = (1.0 - ema_eta) * kp_hat_prev + ema_eta * kp_raw
        kp_hat_t = np.maximum(kp_hat_t, kp_min)
        kp_hat_prev = kp_hat_t.copy()
        kph_seq.append(kp_hat_t.copy())

        # (C) gravity-aware correction step using current Kp estimate
        q_des_ctrl, q_eq_ctrl, info_grav = est.ik_step_gravity(
            q_des=q_des_ctrl, kp_vec=kp_hat_t, T_des=T_des,
            w_pos=w_pos_grav, w_rot=w_rot_grav,
            lambda_damp=lam_grav,
            step_scale=step_scale_grav,
            use_limits=True
        )
        qeq_seq.append(q_eq_ctrl.copy())
        qdes_seq_ctrl.append(q_des_ctrl.copy())
        err_grav_seq.append(info_grav["err_norm"])

        # (D) live visualization with target axes
        ax.cla()
        p_des_arm = est.joint_positions(q_des_ctrl)
        p_eq_arm = est.joint_positions(q_eq_ctrl)
        est.plot_arm_positions(ax, p_des_arm, {"linestyle": "--"}, label="q_des")
        est.plot_arm_positions(ax, p_eq_arm, {"linewidth": 2}, label="q_eq")
        est.draw_frames(ax, q_eq_ctrl, axis_length=0.05)
        est.draw_pose_axes(ax, T_des, axis_length=0.07)

        all_pts = np.vstack([p_des_arm, p_eq_arm, T_des.translation.reshape(1,3)])
        c = all_pts.mean(axis=0)
        r = max(np.max(np.linalg.norm(all_pts - c, axis=1)), 0.2)
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.set_title(f"t={t}: rigid_err={info_rig['err_norm']:.2e}, grav_err={info_grav['err_norm']:.2e}")
        ax.legend(); ax.view_init(elev=25, azim=45); plt.gca().set_box_aspect([1,1,1])
        plt.pause(0.01)

    plt.ioff(); plt.show()

    # report
    kp_hat = np.exp(ukf.x)
    print("Kp_true =", kp_true)
    print("Kp_hat  =", kp_hat)
    rel_err = np.linalg.norm(kp_hat - kp_true) / (np.linalg.norm(kp_true) + 1e-12)
    print("relative_error =", rel_err)

    # final snapshot
    est.visualize_with_target(qdes_seq_ctrl[-1], qeq_seq[-1], T_des,
                              title="Final: alternating rigid-then-gravity correction")
