# Rigid IK (IKPy) + UKF with dense path sampling + Gravity-aware IK
# Two completely separate Pinocchio instances:
#   - est_true: used ONLY to generate measurements (A_t) with Kp_true
#   - est_est : used by UKF and gravity-aware IK (no contamination)
#
# Fixes kept:
# - A_t normalization (spectral) so y in [-1, 0]
# - Larger R for UKF
# - Wide clip on log(Kp)
# - Fallback densification if gravity-IK path is too short
# - Consistency check prints

import os
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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
    T = np.eye(4)
    T[:3, :3] = M.rotation
    T[:3, 3] = M.translation
    return T

# ------------------------ Bingham helpers ------------------------
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

def bingham_A_normalized(g_base_vec, gee_vec, parameter=1.0):
    A = simple_bingham_unit(g_base_vec, gee_vec, parameter=parameter)
    s = np.linalg.norm(A, 2)  # spectral norm
    return A / (s + 1e-12)

# ------------------------ Estimator (one class, two instances) ------------------------
class SixRArm:
    def __init__(self, urdf_path: str, tip_link_name: str = "link6", base_link_name: str = "base_link"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.tip_fid = self.model.getFrameId(tip_link_name)
        self.base_fid = self.model.getFrameId(base_link_name)

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

    # EE orientation (base->ee)
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

    # IK outer loop bits (for est_est only)
    def se3_error(self, T_cur: pin.SE3, T_des: pin.SE3) -> np.ndarray:
        E = T_cur.inverse() * T_des
        dp = E.translation
        dth = pin.log3(E.rotation)
        return np.hstack([dp, dth])

    def frame_jacobian_local(self, q: np.ndarray, frame_id: int) -> np.ndarray:
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        ref = pin.ReferenceFrame.LOCAL if hasattr(pin, "ReferenceFrame") else pin.LOCAL
        return pin.getFrameJacobian(self.model, self.data, frame_id, ref)

# ------------------------ UKF ------------------------
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
        self.last_qeq = None  # warm start (for est_est)

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

    # NOTE: this uses est_est ONLY
    def update(self, z_t: float, q_des_t: np.ndarray, A_t: np.ndarray, est_est: SixRArm):
        sig = self._sigma_points(self.x, self.P)
        y_sig = np.zeros(sig.shape[0])
        qeq_ws = self.last_qeq
        for i, xs in enumerate(sig):
            kp_vec = np.exp(xs)
            q_eq = est_est.calc_equilibrium_s1(q_des_t, kp_vec,
                                               maxiter=self.sigma_maxiter,
                                               q_init=qeq_ws)
            qeq_ws = q_eq  # chain warm start
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

# est_true → A_t（真値）, est_est → UKF 更新（推定）
def update_ukf_on_path(est_true: SixRArm,
                       est_est: SixRArm,
                       ukf: LogKpUKF,
                       kp_true: np.ndarray,
                       g_base: np.ndarray,
                       qdes_path: np.ndarray,
                       newton_iter_true: int = 60):
    qeq_ws_true = None
    for qd in qdes_path:
        # Plant equilibrium (true Kp) on est_true
        q_eq_true = est_true.calc_equilibrium_s1(qd, kp_true, maxiter=newton_iter_true, q_init=qeq_ws_true)
        qeq_ws_true = q_eq_true
        gee_true = est_true.gravity_dir_in_ee(q_eq_true, g_base)
        A_t = bingham_A_normalized(g_base, gee_true, parameter=1.0)  # normalized
        ukf.predict()
        ukf.update(0.0, qd, A_t, est_est)

def check_measurement_consistency(est_true: SixRArm,
                                  g_base: np.ndarray,
                                  kp_true: np.ndarray,
                                  qdes_path: np.ndarray,
                                  n_check: int = 10,
                                  newton_iter: int = 80):
    print("== consistency check (y_true should be ~0) ==")
    idxs = np.linspace(0, len(qdes_path)-1, n_check, dtype=int)
    qeq_ws = None
    for i in idxs:
        qd = qdes_path[i]
        q_eq_true = est_true.calc_equilibrium_s1(qd, kp_true, maxiter=newton_iter, q_init=qeq_ws)
        qeq_ws = q_eq_true
        gee_true = est_true.gravity_dir_in_ee(q_eq_true, g_base)
        A_t = bingham_A_normalized(g_base, gee_true, parameter=1.0)
        q_wxyz_true = est_true.ee_quaternion_wxyz_base(q_eq_true)
        y_true = float(q_wxyz_true.T @ A_t @ q_wxyz_true)
        R_be_true = est_true.ee_rotation_in_base(q_eq_true)
        gb_unit = g_base / (np.linalg.norm(g_base) + 1e-12)
        gpred = R_be_true @ gb_unit
        gpred = gpred / (np.linalg.norm(gpred) + 1e-12)
        cosang = float(np.clip(np.dot(gpred, gee_true), -1.0, 1.0))
        print(f"i={i:03d}  y_true={y_true:+.3e}   cos(gpred,gee_true)={cosang:.6f}")

# Gravity-aware IK on est_est, record path for UKF
def ik_gravity_solve_full_with_path(est_est: SixRArm,
                                    kp_vec: np.ndarray,
                                    q_des_start: np.ndarray,
                                    T_des: pin.SE3,
                                    w_pos: float = 1.0,
                                    w_rot: float = 0.5,
                                    lambda_damp: float = 1e-5,
                                    max_iters: int = 180,
                                    pos_tol: float = 1e-3,
                                    rot_tol: float = 1e-2):
    q_des = q_des_start.copy()
    path = [q_des.copy()]
    fid = est_est.tip_fid
    nv = est_est.nv
    K = np.diag(kp_vec)
    q_eq_ws = None
    for it in range(max_iters):
        q_eq = est_est.calc_equilibrium_s1(q_des, kp_vec, maxiter=80, q_init=q_eq_ws)
        q_eq_ws = q_eq
        pin.forwardKinematics(est_est.model, est_est.data, q_eq)
        pin.updateFramePlacements(est_est.model, est_est.data)
        T_cur = est_est.data.oMf[fid]
        e = est_est.se3_error(T_cur, T_des)
        if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
            return q_des, q_eq, {"iters": it, "converged": True}, np.array(path)
        J6 = est_est.frame_jacobian_local(q_eq, fid)
        G = pin.computeGeneralizedGravityDerivatives(est_est.model, est_est.data, q_eq)
        A = G + K
        B = np.linalg.solve(A, K)
        M = -J6 @ B
        
        W = np.diag([w_pos, w_pos, w_pos, w_rot, w_rot, w_rot])
        H = M.T @ W @ M + lambda_damp * np.eye(nv)
        g = M.T @ W @ e
        dq_des = -np.linalg.solve(H, g)
        q_des = q_des + dq_des
        if hasattr(est_est.model, "lowerPositionLimit"):
            lo = est_est.model.lowerPositionLimit; hi = est_est.model.upperPositionLimit
            q_des = np.minimum(np.maximum(q_des, lo), hi)
        path.append(q_des.copy())
    q_eq = est_est.calc_equilibrium_s1(q_des, kp_vec, maxiter=80, q_init=q_eq_ws)
    return q_des, q_eq, {"iters": max_iters, "converged": False}, np.array(path)

# ------------------------ Visuals (optional) ------------------------
def draw_pose_axes(ax, T: pin.SE3, axis_length: float = 0.07):
    colors = ["r", "g", "b"]; o = T.translation; Rm = T.rotation
    for k, c in enumerate(colors):
        a = Rm[:, k] * axis_length
        ax.plot([o[0], o[0]+a[0]], [o[1], o[1]+a[1]], [o[2], o[2]+a[2]], c, linewidth=2.0)
    ax.scatter([o[0]], [o[1]], [o[2]], s=80, c="k", marker="*", label="target")

def joint_positions(est: SixRArm, q: np.ndarray) -> np.ndarray:
    pin.forwardKinematics(est.model, est.data, q)
    pin.updateFramePlacements(est.model, est.data)
    pts = []
    for jid in range(1, est.model.njoints):
        pts.append(est.data.oMi[jid].translation)
    pts.append(est.data.oMf[est.tip_fid].translation)
    return np.vstack(pts)

def visualize_with_target(est: SixRArm, q_des: np.ndarray, q_eq: np.ndarray, T_des: pin.SE3, title: str):
    p_des = joint_positions(est, q_des)
    p_eq = joint_positions(est, q_eq)
    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(p_des[:,0], p_des[:,1], p_des[:,2], "o--", label="q_des")
    ax.plot(p_eq[:,0],  p_eq[:,1],  p_eq[:,2],  "o-",  label="q_eq")
    draw_pose_axes(ax, T_des, 0.07)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title(title); ax.legend()
    all_pts = np.vstack([p_des, p_eq, T_des.translation.reshape(1,3)])
    c = all_pts.mean(axis=0); r = max(np.max(np.linalg.norm(all_pts-c, axis=1)), 0.2)
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.view_init(elev=25, azim=45); plt.gca().set_box_aspect([1,1,1]); plt.tight_layout(); plt.show()

# ------------------------ Main ------------------------
if __name__ == "__main__":
    urdf_path = ensure_urdf()

    # Separate instances
    est_true = SixRArm(urdf_path, tip_link_name="link6", base_link_name="base_link")  # plant-only
    est_est  = SixRArm(urdf_path, tip_link_name="link6", base_link_name="base_link")  # filter/IK-only

    # Plant (unknown to estimator)
    kp_true = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float)
    g_base = np.array([0.0, 0.0, -9.8])

    # IKPy chain and reachable target by FK (once)
    chain = IkChain.from_urdf_file(urdf_path, base_elements=["base_link"], symbolic=False)
    model = est_true.model; data = est_true.data
    rng = np.random.default_rng(42)
    q_ref = np.array([rng.uniform(lo, hi) for lo, hi in zip(model.lowerPositionLimit, model.upperPositionLimit)])
    pin.forwardKinematics(model, data, q_ref)
    jid = model.getJointId("joint6")
    oM_des = pin.SE3(data.oMi[jid])  # copy
    T_target = se3_to_homog(oM_des)
    T_des = pin.SE3(T_target[:3, :3], T_target[:3, 3])

    # Initial desired
    q_des = np.zeros(est_est.nv)

    # UKF init (normalized y)
    x0 = np.log(np.ones(est_est.nv) * 50.0)
    P0 = np.eye(est_est.nv) * 1.0
    Q = np.eye(est_est.nv) * 1e-3
    R_scalar = 1e-2
    ukf = LogKpUKF(x0, P0, Q, R_scalar, alpha=1e-2, beta=2.0, kappa=0.0,
                   kp_clip=(1e-3, 1e3), sigma_maxiter=80)

    # ---------- Phase 1: Rigid IK (IKPy) ----------
    q0_full = np.zeros(len(chain.links))
    q_sol_full = chain.inverse_kinematics_frame(
        T_target, initial_position=q0_full, max_iter=1000, orientation_mode="all"
    )
    q_sol_rigid = np.array(q_sol_full[1:1+est_est.nv])

    n_path = 80
    q_path = make_joint_path(q_des, q_sol_rigid, n_path)
    # check_measurement_consistency(est_true, g_base, kp_true, q_path, n_check=10, newton_iter=60)
    update_ukf_on_path(est_true, est_est, ukf, kp_true, g_base, q_path, newton_iter_true=60)

    kp_phase = np.maximum(np.exp(ukf.x), 1e-6)
    q_des = q_sol_rigid.copy()
    q_eq_vis = est_true.calc_equilibrium_s1(q_des, kp_true, maxiter=80)  # for visualization only (truth)
    visualize_with_target(est_true, q_des, q_eq_vis, T_des, title="After Phase 1 (rigid IK, truth pose)")

    # ---------- Phase 2..N: Gravity-aware IK on est_est ----------
    num_gravity_passes = 3
    for i in range(num_gravity_passes):
        print("PHASE 2")
        q_des_new, q_eq_new, info, q_path_phase = ik_gravity_solve_full_with_path(
            est_est, kp_phase, q_des, T_des,
            w_pos=1.0, w_rot=0.5, lambda_damp=1e-5,
            max_iters=180, pos_tol=1e-3, rot_tol=1e-2
        )
        # densify if path too short
        if q_path_phase.shape[0] < 2:
            n_fb = 20; al = np.linspace(0.0, 1.0, n_fb)
            q_path_phase = (1-al)[:,None]*q_des[None,:] + al[:,None]*q_des_new[None,:]

        # check_measurement_consistency(est_true, g_base, kp_true, q_path_phase, n_check=10, newton_iter=50)
        update_ukf_on_path(est_true, est_est, ukf, kp_true, g_base, q_path_phase, newton_iter_true=50)

        kp_phase = np.maximum(np.exp(ukf.x), 1e-6)
        q_des = q_des_new.copy()
        visualize_with_target(est_est, q_des, q_eq_new, T_des, title=f"After Phase 2.{i+1} (gravity IK, est pose)")

    # ---------- Report ----------
    print("Kp_true =", kp_true)
    kp_hat_final = np.exp(ukf.x)
    print("Kp_hat  =", kp_hat_final)
    rel_err = np.linalg.norm(kp_hat_final - kp_true) / (np.linalg.norm(kp_true) + 1e-12)
    print("relative_error =", rel_err)
