# IKPy-based gravity-aware reaching + online UKF of Kp (A not normalized, R=1e-1)
# - IK is always solved by IKPy (rigid). Gravity is handled by an outer fixed-point loop.
# - Two Pinocchio instances: est_true (measurements with Kp_true) / est_est (UKF & equilibrium & viz).
# - UKF uses y = q(wxyz)^T A q(wxyz), with A built from true gravity direction (NO normalization).
# - Final viz shows: rigid(q_cmd), true-grav(q_eq_true), est-grav(q_eq_est with Kp used in IK).

import os
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc, Slerp
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from ikpy.chain import Chain as IkChain

# ------------------------ URDF ------------------------
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

def se3_to_homog(M: pin.SE3) -> np.ndarray:
    T = np.eye(4); T[:3, :3] = M.rotation; T[:3, 3] = M.translation
    return T

# ------------------------ Bingham helpers (NO normalization) ------------------------
class Quaternion:
    def __init__(self, w, x, y, z):
        self._array = np.array([w, x, y, z], dtype=float)
        self.w = w; self.x = x; self.y = y; self.z = z
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

def simple_bingham_unit(before_vec3: np.ndarray, after_vec3: np.ndarray, parameter: float = 100.0) -> np.ndarray:
    # parameter is the only scale knob (NO normalization)
    b = np.asarray(before_vec3, dtype=float)
    a = np.asarray(after_vec3, dtype=float)
    bn = b / (np.linalg.norm(b) + 1e-12)
    an = a / (np.linalg.norm(a) + 1e-12)
    vq = Quaternion(0.0, bn[0], bn[1], bn[2])
    xq = Quaternion(0.0, an[0], an[1], an[2])
    P = Lmat(xq) - Rmat(vq)
    A0 = -0.25 * (P.T @ P)   # negative semidefinite
    return float(parameter) * A0

# ------------------------ Robot wrapper ------------------------
class SixRArm:
    def __init__(self, urdf_path: str, tip_link_name: str = "link6", base_link_name: str = "base_link"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.tip_fid = self.model.getFrameId(tip_link_name)
        self.base_fid = self.model.getFrameId(base_link_name)
        if hasattr(self.model, "gravity"):
            self.model.gravity.linear = np.array([0.0, 0.0, -9.81])

    # S^1-Newton equilibrium
    def calc_equilibrium_s1(self, q_des: np.ndarray, kp_vec: np.ndarray,
                            maxiter: int = 80, eps: float = 1e-12,
                            q_init: Optional[np.ndarray] = None) -> np.ndarray:
        K = np.diag(kp_vec)
        if q_init is None:
            cos_q = np.cos(q_des.copy()); sin_q = np.sin(q_des.copy())
        else:
            cos_q = np.cos(q_init.copy()); sin_q = np.sin(q_init.copy())
        for _ in range(maxiter):
            q = np.arctan2(sin_q, cos_q)
            F = pin.computeGeneralizedGravity(self.model, self.data, q) + K.dot(q - q_des)
            dG = pin.computeGeneralizedGravityDerivatives(self.model, self.data, q)
            dF = dG + K
            n = self.nv
            D = np.zeros((n, 2*n))
            denom = cos_q**2 + sin_q**2
            D[np.arange(n), 2*np.arange(n)] = -sin_q / denom
            D[np.arange(n), 2*np.arange(n)+1] =  cos_q / denom
            J = dF.dot(D)
            dq = np.linalg.pinv(J).dot(F)
            cu = cos_q - dq[0::2]; su = sin_q - dq[1::2]
            nrm = np.sqrt(cu**2 + su**2)
            cos_q, sin_q = cu/nrm, su/nrm
            if np.linalg.norm(F) < eps:
                break
        return np.arctan2(sin_q, cos_q)

    def ee_rotation_in_base(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        R_wb = self.data.oMf[self.base_fid].rotation
        R_we = self.data.oMf[self.tip_fid].rotation
        return R_wb.T @ R_we

    def ee_quaternion_wxyz_base(self, q: np.ndarray) -> np.ndarray:
        R_be = self.ee_rotation_in_base(q)
        q_xyzw = Rsc.from_matrix(R_be).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        return q_wxyz / np.linalg.norm(q_wxyz)

    def gravity_dir_in_ee(self, q: np.ndarray, g_base: np.ndarray) -> np.ndarray:
        R_be = self.ee_rotation_in_base(q)
        gb = g_base / (np.linalg.norm(g_base) + 1e-12)
        gee = R_be @ gb
        return gee / (np.linalg.norm(gee) + 1e-12)

# ------------------------ UKF (state = log Kp) ------------------------
class LogKpUKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R_scalar: float,
                 alpha: float = 1e-2, beta: float = 2.0, kappa: float = 0.0,
                 kp_clip: Optional[Tuple[float, float]] = (1e-3, 1e3),
                 sigma_maxiter: int = 80):
        self.x = x0.copy(); self.P = P0.copy(); self.Q = Q.copy()
        self.R = float(R_scalar); self.n = x0.size
        self.alpha = alpha; self.beta = beta; self.kappa = kappa
        self.lmbda = alpha**2 * (self.n + kappa) - self.n
        self.c = self.n + self.lmbda
        self.Wm = np.full(2*self.n+1, 1.0/(2.0*self.c))
        self.Wc = np.full(2*self.n+1, 1.0/(2.0*self.c))
        self.Wm[0] = self.lmbda/self.c
        self.Wc[0] = self.lmbda/self.c + (1.0 - alpha**2 + beta)
        self.kp_clip = kp_clip
        self.sigma_maxiter = sigma_maxiter
        self.last_qeq = None

    def _sigma_points(self, x, P):
        n = self.n
        Psym = 0.5*(P+P.T) + 1e-10*np.eye(n)
        try:
            S = np.linalg.cholesky(self.c * Psym)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky(self.c * (Psym + 1e-8*np.eye(n)))
        sig = np.zeros((2*n+1, n)); sig[0] = x
        for i in range(n):
            col = S[:, i]
            sig[1+i] = x + col
            sig[1+n+i] = x - col
        return sig

    def predict(self):
        self.P = self.P + self.Q

    def update(self, z_t: float, q_des_t: np.ndarray, A_t: np.ndarray, est_est: SixRArm):
        sig = self._sigma_points(self.x, self.P)
        y_sig = np.zeros(sig.shape[0])
        qeq_ws = self.last_qeq
        for i, xs in enumerate(sig):
            kp_vec = np.exp(xs)
            q_eq = est_est.calc_equilibrium_s1(q_des_t, kp_vec,
                                               maxiter=self.sigma_maxiter,
                                               q_init=qeq_ws)
            qeq_ws = q_eq
            q_wxyz = est_est.ee_quaternion_wxyz_base(q_eq)
            y_sig[i] = float(q_wxyz.T @ A_t @ q_wxyz)

        y_pred = np.dot(self.Wm, y_sig)
        dy = y_sig - y_pred
        S = float(np.dot(self.Wc, dy * dy)) + self.R
        C = np.sum(self.Wc[:, None] * (sig - self.x) * dy[:, None], axis=0).reshape(self.n, 1)
        K = C / (S + 1e-18)

        self.x = self.x + (K.flatten() * (z_t - y_pred))
        if self.kp_clip is not None:
            lo, hi = self.kp_clip
            self.x = np.clip(self.x, np.log(lo), np.log(hi))
        self.P = self.P - K @ K.T * S
        self.P = 0.5 * (self.P + self.P.T)
        self.last_qeq = qeq_ws

# ------------------------ Helpers ------------------------
def make_joint_path(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    al = np.linspace(0.0, 1.0, n)
    return (1 - al)[:, None] * q0[None, :] + al[:, None] * q1[None, :]

def se3_intermediate_targets(T_start: pin.SE3, T_goal: pin.SE3, n_segments: int):
    p0 = T_start.translation; p1 = T_goal.translation
    R0 = Rsc.from_matrix(T_start.rotation)
    R1 = Rsc.from_matrix(T_goal.rotation)
    key_times = [0.0, 1.0]
    key_rots = Rsc.from_quat(np.vstack([R0.as_quat(), R1.as_quat()]))
    slerp = Slerp(key_times, key_rots)
    ts = np.linspace(0.0, 1.0, n_segments+1)[1:]
    rots = slerp(ts)
    targets = []
    for s, Rsi in zip(ts, rots.as_matrix()):
        pi = (1 - s) * p0 + s * p1
        targets.append(pin.SE3(Rsi, pi))
    return targets

# Build A_t from true gravity direction (NO normalization)
def build_A_from_true(est_true: SixRArm, q_des: np.ndarray, kp_true: np.ndarray, g_base: np.ndarray,
                      parameter_A: float = 100.0, newton_iter_true: int = 60,
                      qeq_ws_true: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    q_eq_true = est_true.calc_equilibrium_s1(q_des, kp_true, maxiter=newton_iter_true, q_init=qeq_ws_true)
    gee_true = est_true.gravity_dir_in_ee(q_eq_true, g_base)
    A_t = simple_bingham_unit(g_base, gee_true, parameter=parameter_A)
    return A_t, q_eq_true

# est_true → A_t（真値），est_est → UKF 更新
def update_ukf_on_path(est_true: SixRArm, est_est: SixRArm, ukf: LogKpUKF,
                       kp_true: np.ndarray, g_base: np.ndarray,
                       qdes_path: np.ndarray, parameter_A: float = 100.0,
                       newton_iter_true: int = 60):
    qeq_ws_true = None
    for qd in qdes_path:
        A_t, q_eq_true = build_A_from_true(est_true, qd, kp_true, g_base,
                                           parameter_A=parameter_A,
                                           newton_iter_true=newton_iter_true,
                                           qeq_ws_true=qeq_ws_true)
        qeq_ws_true = q_eq_true
        ukf.predict()
        ukf.update(0.0, qd, A_t, est_est)

# ------------------------ IKPy-based gravity-aware IK (outer loop) ------------------------
def ikpy_inverse(chain: IkChain, T_target: np.ndarray, q_init_joint: np.ndarray,
                 max_iter: int = 800) -> np.ndarray:
    q0_full = np.zeros(len(chain.links))
    q0_full[1:1+q_init_joint.size] = q_init_joint
    q_sol_full = chain.inverse_kinematics_frame(
        T_target, initial_position=q0_full, max_iter=max_iter, orientation_mode="all"
    )
    return np.array(q_sol_full[1:1+q_init_joint.size])

def gravity_aware_ik_with_ikpy(est_est: SixRArm, chain: IkChain,
                               kp_vec: np.ndarray, q_des_start: np.ndarray, T_des: pin.SE3,
                               max_outer_iters: int = 8, densify_steps: int = 60,
                               pos_tol: float = 1e-3, rot_tol: float = 1e-2,
                               kp_rigid_thresh: float = 500.0, alpha_step: float = 1.0):
    """
    Outer fixed-point: use IKPy (rigid) to update q_des so that equilibrium(q_des, Kp) reaches T_des.
    Returns: q_des_final, q_eq_est_final, info, q_path (densified for UKF)
    """
    T_target = se3_to_homog(T_des)
    q_des = q_des_start.copy()
    q_eq_ws = None

    # Rigid short-circuit: if Kp is huge, just one IKPy solve
    if float(np.min(kp_vec)) >= kp_rigid_thresh:
        q_rigid = ikpy_inverse(chain, T_target, q_des, max_iter=1200)
        q_path = make_joint_path(q_des, q_rigid, max(2, densify_steps))
        return q_rigid, q_rigid, {"iters": 1, "converged": True, "rigid_mode": True}, q_path

    # General case: fixed-point loop
    path_knots = [q_des.copy()]
    for it in range(max_outer_iters):
        # equilibrium under current Kp
        q_eq = est_est.calc_equilibrium_s1(q_des, kp_vec, maxiter=80, q_init=q_eq_ws)
        q_eq_ws = q_eq
        # pose error at equilibrium
        pin.forwardKinematics(est_est.model, est_est.data, q_eq)
        pin.updateFramePlacements(est_est.model, est_est.data)
        T_cur = est_est.data.oMf[est_est.tip_fid]
        E = T_cur.inverse() * T_des
        dp = E.translation; dth = pin.log3(E.rotation)

        if np.linalg.norm(dp) < pos_tol and np.linalg.norm(dth) < rot_tol:
            # densify last leg
            q_path = make_joint_path(path_knots[-1], q_des, max(2, densify_steps))
            return q_des, q_eq, {"iters": it, "converged": True, "rigid_mode": False}, q_path

        # rigid IK step (ikpy) with current q_des as init
        q_rigid = ikpy_inverse(chain, T_target, q_des, max_iter=800)
        # relaxed update (alpha=1.0 by default)
        q_new = (1.0 - alpha_step) * q_des + alpha_step * q_rigid
        path_knots.append(q_new.copy())
        q_des = q_new

    # not converged: return last
    q_path = []
    for a, b in zip(path_knots[:-1], path_knots[1:]):
        q_path.append(make_joint_path(a, b, max(2, densify_steps)))
    q_path = np.vstack(q_path) if len(q_path) > 0 else make_joint_path(q_des_start, q_des, max(2, densify_steps))
    q_eq = est_est.calc_equilibrium_s1(q_des, kp_vec, maxiter=80, q_init=q_eq_ws)
    return q_des, q_eq, {"iters": max_outer_iters, "converged": False, "rigid_mode": False}, q_path

# ------------------------ Visualization ------------------------
def joint_positions_generic(est: SixRArm, q: np.ndarray) -> np.ndarray:
    pin.forwardKinematics(est.model, est.data, q)
    pin.updateFramePlacements(est.model, est.data)
    pts = []
    for jid in range(1, est.model.njoints):
        pts.append(est.data.oMi[jid].translation)
    pts.append(est.data.oMf[est.tip_fid].translation)
    return np.vstack(pts)

def draw_pose_axes(ax, T: pin.SE3, axis_length: float = 0.07):
    o = T.translation; Rm = T.rotation
    for k, c in enumerate(["r","g","b"]):
        a = Rm[:,k]*axis_length
        ax.plot([o[0], o[0]+a[0]], [o[1], o[1]+a[1]], [o[2], o[2]+a[2]], c, linewidth=2.0)
    ax.scatter([o[0]], [o[1]], [o[2]], s=80, c="k", marker="*", label="target")

def visualize_triplet(est_true: SixRArm, est_est: SixRArm,
                      q_cmd: np.ndarray, kp_true: np.ndarray, kp_used_in_ik: np.ndarray,
                      T_des: pin.SE3, title: str):
    P_rigid = joint_positions_generic(est_true, q_cmd)
    q_eq_true = est_true.calc_equilibrium_s1(q_cmd, kp_true, maxiter=80)
    P_true = joint_positions_generic(est_true, q_eq_true)
    q_eq_est = est_est.calc_equilibrium_s1(q_cmd, np.maximum(kp_used_in_ik,1e-6), maxiter=80)
    P_est = joint_positions_generic(est_est, q_eq_est)

    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(P_rigid[:,0], P_rigid[:,1], P_rigid[:,2], "o--",  label="rigid (q_cmd)")
    ax.plot(P_true[:,0],  P_true[:,1],  P_true[:,2],  "o-",   label="grav true Kp")
    ax.plot(P_est[:,0],   P_est[:,1],   P_est[:,2],   "o-.",  label="grav est  Kp (IK-time)")
    draw_pose_axes(ax, T_des, axis_length=0.07)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title(title); ax.legend()
    all_pts = np.vstack([P_rigid, P_true, P_est, T_des.translation.reshape(1,3)])
    c = all_pts.mean(axis=0); r = max(np.max(np.linalg.norm(all_pts-c, axis=1)), 0.2)
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.view_init(elev=25, azim=45); plt.gca().set_box_aspect([1,1,1]); plt.tight_layout(); plt.show()

# ------------------------ Main ------------------------
if __name__ == "__main__":
    urdf_path = ensure_urdf()

    # Separate instances
    est_true = SixRArm(urdf_path, tip_link_name="link6", base_link_name="base_link")  # plant-only
    est_est  = SixRArm(urdf_path, tip_link_name="link6", base_link_name="base_link")  # filter/IK-only

    # IKPy chain
    chain = IkChain.from_urdf_file(urdf_path, base_elements=["base_link"], symbolic=False)

    # Plant truth
    kp_true = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float)
    g_base = np.array([0.0, 0.0, -9.81])
    parameter_A = 100.0  # scale for A (NO normalization)

    # Make reachable target (FK from random q_ref, TIP frame)
    model = est_true.model; data = est_true.data
    rng = np.random.default_rng(8)
    q_ref = np.array([rng.uniform(lo, hi) for lo, hi in zip(model.lowerPositionLimit, model.upperPositionLimit)])
    pin.forwardKinematics(model, data, q_ref); pin.updateFramePlacements(model, data)
    fid_tip = model.getFrameId("link6")
    oM_des = pin.SE3(data.oMf[fid_tip]); T_des = oM_des

    # UKF init (R=1e-1 as requested)
    q_des = np.zeros(est_est.nv)
    x0 = np.log(np.ones(est_est.nv) * 25.0)
    P0 = np.eye(est_est.nv) * 1.0
    Q = np.eye(est_est.nv) * 1e-3
    R_scalar = 1e-1
    ukf = LogKpUKF(x0, P0, Q, R_scalar, alpha=1e-2, beta=2.0, kappa=0.0,
                   kp_clip=(1e-3, 1e3), sigma_maxiter=80)
    kp_hat = np.maximum(np.exp(ukf.x), 1e-6)

    # Build segment targets (you can set n_segments=1)
    q_eq_start = est_est.calc_equilibrium_s1(q_des, kp_hat, maxiter=80)
    pin.forwardKinematics(est_est.model, est_est.data, q_eq_start); pin.updateFramePlacements(est_est.model, est_est.data)
    T_start = est_est.data.oMf[est_est.tip_fid]
    n_segments = 4
    segment_targets = se3_intermediate_targets(T_start, T_des, n_segments)

    kp_used_hist = []
    q_cmd_hist   = []

    for i, T_seg in enumerate(segment_targets, 1):
        kp_used = kp_hat.copy()  # Kp used in this segment's IK (for consistent viz)
        q_des_new, q_eq_est, info, q_path = gravity_aware_ik_with_ikpy(
            est_est, chain, kp_used, q_des, T_seg,
            max_outer_iters=8, densify_steps=60,
            pos_tol=1e-3, rot_tol=1e-2,
            kp_rigid_thresh=500.0, alpha_step=1.0
        )
        print(f"[Segment {i}/{n_segments}] info = {info}")

        # record for triplet viz
        kp_used_hist.append(kp_used)
        q_cmd_hist.append(q_des_new.copy())

        # UKF update along the traversed q_des path
        update_ukf_on_path(est_true, est_est, ukf, kp_true, g_base, q_path,
                           parameter_A=parameter_A, newton_iter_true=60)

        kp_hat = np.maximum(np.exp(ukf.x), 1e-6)
        q_des = q_des_new.copy()
        print(f"[Segment {i}/{n_segments}]  Kp_hat =", kp_hat)

    # Final triplet viz (use the Kp that was actually used in the last IK)
    q_cmd_final   = q_cmd_hist[-1]
    kp_used_final = kp_used_hist[-1]
    visualize_triplet(est_true, est_est, q_cmd_final, kp_true, kp_used_final, T_des,
                      title="Triplet: rigid vs true-gravity vs est-gravity (IKPy-based)")

    # Report
    print("Kp_true =", kp_true)
    kp_hat_final = np.exp(ukf.x)
    print("Kp_hat  =", kp_hat_final)
    rel_err = np.linalg.norm(kp_hat_final - kp_true) / (np.linalg.norm(kp_true) + 1e-12)
    print("relative_error =", rel_err)
