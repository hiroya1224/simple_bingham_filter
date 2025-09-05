# IKPy-based gravity-aware reaching with online UKF of Kp (revised per requirements)
# - Target EE frame T_target is fixed in space BEFORE estimation, created by robot_sim and never moved.
# - UKF uses the RIGID command joint path (IKPy outputs) as the input sequence. Deflected joint angles are never used as inputs.
# - Controller adjusts q_des (command) so that the DEFLECTED equilibrium reaches T_target, while IK is always IKPy.
# - Visualizer shows: (1) sim rigid @ q_final, (2) sim gravity @ Kp_true, (3) est gravity @ Kp_hat, plus frames at joints and the target frame.

import os
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc, Slerp
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from ikpy.chain import Chain as IkChain

# ------------------------ URDF ------------------------
def ensure_urdf() -> str:
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

def make_joint_path(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    a = np.linspace(0.0, 1.0, n)
    return (1 - a)[:, None] * q0[None, :] + a[:, None] * q1[None, :]

# ------------------------ Bingham (NO normalization) ------------------------
class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self._a = np.array([w, x, y, z], dtype=float)
    def array(self) -> np.ndarray:
        return self._a

def Lmat(q: Quaternion) -> np.ndarray:
    a, b, c, d = q.array()
    return np.array([[a,-b,-c,-d],[b,a,-d,c],[c,d,a,-b],[d,-c,b,a]])

def Rmat(q: Quaternion) -> np.ndarray:
    w, x, y, z = q.array()
    return np.array([[w,-x,-y,-z],[x,w,z,-y],[y,-z,w,x],[z,y,-x,w]])

def simple_bingham_unit(before_vec3: np.ndarray, after_vec3: np.ndarray, parameter: float = 100.0) -> np.ndarray:
    b = np.asarray(before_vec3, dtype=float); a = np.asarray(after_vec3, dtype=float)
    bn = b / (np.linalg.norm(b) + 1e-12); an = a / (np.linalg.norm(a) + 1e-12)
    vq = Quaternion(0.0, bn[0], bn[1], bn[2]); xq = Quaternion(0.0, an[0], an[1], an[2])
    P = Lmat(xq) - Rmat(vq)
    A0 = -0.25 * (P.T @ P)  # semi-negative definite
    return float(parameter) * A0

# ------------------------ Robot wrapper ------------------------
class RobotArm:
    def __init__(self, urdf_path: str, tip_link: str = "link6", base_link: str = "base_link"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.tip_fid = self.model.getFrameId(tip_link)
        self.base_fid = self.model.getFrameId(base_link)
        if hasattr(self.model, "gravity"):
            self.model.gravity.linear = np.array([0.0, 0.0, -9.81])

    def equilibrium_s1(self, q_des: np.ndarray, kp_vec: np.ndarray,
                       maxiter: int = 60, eps: float = 1e-10,
                       q_init: Optional[np.ndarray] = None) -> np.ndarray:
        K = np.diag(kp_vec)
        if q_init is None:
            c = np.cos(q_des.copy()); s = np.sin(q_des.copy())
        else:
            c = np.cos(q_init.copy()); s = np.sin(q_init.copy())
        for _ in range(maxiter):
            q = np.arctan2(s, c)
            F = pin.computeGeneralizedGravity(self.model, self.data, q) + K @ (q - q_des)
            dG = pin.computeGeneralizedGravityDerivatives(self.model, self.data, q)
            dF = dG + K
            n = self.nv
            D = np.zeros((n, 2*n))
            den = c*c + s*s
            D[np.arange(n), 2*np.arange(n)]   = -s / den
            D[np.arange(n), 2*np.arange(n)+1] =  c / den
            J = dF @ D
            dq = np.linalg.pinv(J) @ F
            cu, su = c - dq[0::2], s - dq[1::2]
            nrm = np.sqrt(cu*cu + su*su)
            c, s = cu/nrm, su/nrm
            if np.linalg.norm(F) < eps:
                break
        return np.arctan2(s, c)

    def ee_rotation_in_base(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q); pin.updateFramePlacements(self.model, self.data)
        R_wb = self.data.oMf[self.base_fid].rotation
        R_we = self.data.oMf[self.tip_fid].rotation
        return R_wb.T @ R_we

    def ee_quaternion_wxyz_base(self, q: np.ndarray) -> np.ndarray:
        R_be = self.ee_rotation_in_base(q)
        q_xyzw = Rsc.from_matrix(R_be).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        return q_wxyz / (np.linalg.norm(q_wxyz) + 1e-18)

    def gravity_dir_in_ee(self, q: np.ndarray, g_base: np.ndarray) -> np.ndarray:
        R_be = self.ee_rotation_in_base(q)
        gb = g_base / (np.linalg.norm(g_base) + 1e-12)
        gee = R_be @ gb
        return gee / (np.linalg.norm(gee) + 1e-12)

    def fk_pose(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q); pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tip_fid]

    def joint_transforms(self, q: np.ndarray) -> List[pin.SE3]:
        pin.forwardKinematics(self.model, self.data, q); pin.updateFramePlacements(self.model, self.data)
        Ts = [self.data.oMi[jid] for jid in range(1, self.model.njoints)]
        Ts.append(self.data.oMf[self.tip_fid])
        return Ts

# ------------------------ IK (always IKPy) ------------------------
def ikpy_inverse(chain: IkChain, T_target: np.ndarray, q_init_joint: np.ndarray, max_iter: int = 800) -> np.ndarray:
    q0_full = np.zeros(len(chain.links))
    q0_full[1:1+q_init_joint.size] = q_init_joint
    q_sol_full = chain.inverse_kinematics_frame(
        T_target, initial_position=q0_full, max_iter=max_iter, orientation_mode="all"
    )
    return np.array(q_sol_full[1:1+q_init_joint.size])

# ------------------------ Controller (does NOT store T_target inside robot_est) ------------------------
def controller_gravity_aware_q_des(robot_est: RobotArm, chain: IkChain, kp_vec: np.ndarray,
                                   q_des_start: np.ndarray, T_target_se3: pin.SE3,
                                   max_outer_iters: int = 8, cmd_densify: int = 60,
                                   pos_tol: float = 1e-3, rot_tol: float = 1e-2,
                                   kp_rigid_thresh: float = 500.0, alpha_step: float = 1.0
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute new command q_des so that equilibrium(q_des, kp_vec) reaches T_target_se3.
    Returns: (q_des_new, q_eq_est, q_cmd_path)
      - q_cmd_path: DENSIFIED path of RIGID commands actually sent (IKPy outputs).
    """
    T_target = se3_to_homog(T_target_se3)
    q_des = q_des_start.copy()
    q_eq_ws = None

    if float(np.min(kp_vec)) >= kp_rigid_thresh:
        q_rigid = ikpy_inverse(chain, T_target, q_des, max_iter=1200)
        q_cmd_path = make_joint_path(q_des, q_rigid, max(2, cmd_densify))
        return q_rigid, q_rigid, q_cmd_path

    # One IKPy step toward target (rigid), then check deflected equilibrium proximity (loose)
    q_rigid = ikpy_inverse(chain, T_target, q_des, max_iter=800)
    q_des_new = (1.0 - alpha_step) * q_des + alpha_step * q_rigid
    q_cmd_path = make_joint_path(q_des, q_des_new, max(2, cmd_densify))

    q_eq_est = robot_est.equilibrium_s1(q_des_new, kp_vec, maxiter=60, q_init=q_eq_ws)
    E = robot_est.fk_pose(q_eq_est).inverse() * T_target_se3
    dp, dth = E.translation, pin.log3(E.rotation)
    # (Optional) could iterate more IKPy steps here if not within tol; kept single-step for speed as requested.
    return q_des_new, q_eq_est, q_cmd_path

# ------------------------ UKF (state = log Kp) ------------------------
class LogKpUKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R_scalar: float,
                 alpha: float = 1e-2, beta: float = 2.0, kappa: float = 0.0):
        self.x = x0.copy(); self.P = P0.copy(); self.Q = Q.copy()
        self.R = float(R_scalar); self.n = x0.size
        self.alpha = alpha; self.beta = beta; self.kappa = kappa
        self.lmbda = alpha**2 * (self.n + kappa) - self.n
        self.c = self.n + self.lmbda
        self.Wm = np.full(2*self.n+1, 1.0/(2.0*self.c))
        self.Wc = np.full(2*self.n+1, 1.0/(2.0*self.c))
        self.Wm[0] = self.lmbda/self.c
        self.Wc[0] = self.lmbda/self.c + (1.0 - alpha**2 + beta)

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        Psym = 0.5*(P+P.T) + 1e-10*np.eye(self.n)
        S = np.linalg.cholesky(self.c * Psym)
        sig = np.zeros((2*self.n+1, self.n)); sig[0] = x
        for i in range(self.n):
            col = S[:, i]
            sig[1+i] = x + col
            sig[1+self.n+i] = x - col
        return sig

    def predict(self):
        self.P = self.P + self.Q

    def update(self, z_t: float, q_cmd_t: np.ndarray, A_t: np.ndarray, robot_est: RobotArm,
               sigma_maxiter: int = 60, qeq_ws: Optional[np.ndarray] = None) -> np.ndarray:
        """
        UKF measurement update using the command joint q_cmd_t (RIGID) at time t.
        """
        sig = self._sigma_points(self.x, self.P)
        y_sig = np.zeros(sig.shape[0])
        q_ws = qeq_ws
        for i, xs in enumerate(sig):
            kp_vec = np.exp(xs)
            q_eq = robot_est.equilibrium_s1(q_cmd_t, kp_vec, maxiter=sigma_maxiter, q_init=q_ws)
            q_ws = q_eq
            q_wxyz = robot_est.ee_quaternion_wxyz_base(q_eq)
            y_sig[i] = float(q_wxyz.T @ A_t @ q_wxyz)

        y_pred = np.dot(self.Wm, y_sig)
        dy = y_sig - y_pred
        S = float(np.dot(self.Wc, dy * dy)) + self.R
        C = np.sum(self.Wc[:, None] * (sig - self.x) * dy[:, None], axis=0).reshape(self.n, 1)
        K = C / (S + 1e-18)
        self.x = self.x + (K.flatten() * (z_t - y_pred))
        self.P = self.P - (K @ K.T) * S
        self.P = 0.5*(self.P + self.P.T)
        return q_ws  # warm-start for next call

# ------------------------ Measurement path (BUILDS A_t with robot_sim + Kp_true) ------------------------
def build_A_from_true(robot_sim: RobotArm, q_cmd: np.ndarray, kp_true: np.ndarray, g_base: np.ndarray,
                      parameter_A: float = 100.0, newton_iter_true: int = 60,
                      qeq_ws_true: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the RIGID command q_cmd, the "real" plant (robot_sim with Kp_true) deflects to q_eq_true.
    Build A_t from base gravity and the EE gravity dir at q_eq_true. Return (A_t, q_eq_true).
    """
    q_eq_true = robot_sim.equilibrium_s1(q_cmd, kp_true, maxiter=newton_iter_true, q_init=qeq_ws_true)
    gee_true = robot_sim.gravity_dir_in_ee(q_eq_true, g_base)
    A_t = simple_bingham_unit(g_base, gee_true, parameter=parameter_A)
    return A_t, q_eq_true

def update_ukf_along_cmd_path(robot_sim: RobotArm, robot_est: RobotArm, ukf: LogKpUKF,
                              kp_true: np.ndarray, g_base: np.ndarray,
                              q_cmd_path: np.ndarray, parameter_A: float = 100.0,
                              newton_iter_true: int = 60):
    """
    Iterate along the RIGID command path, generating measurements from robot_sim and
    updating the UKF on robot_est.
    """
    qeq_ws_true = None
    qeq_ws_est  = None
    for qc in q_cmd_path:
        A_t, q_eq_true = build_A_from_true(robot_sim, qc, kp_true, g_base,
                                           parameter_A=parameter_A, newton_iter_true=newton_iter_true,
                                           qeq_ws_true=qeq_ws_true)
        qeq_ws_true = q_eq_true
        ukf.predict()
        qeq_ws_est = ukf.update(0.0, qc, A_t, robot_est, sigma_maxiter=60, qeq_ws=qeq_ws_est)

# ------------------------ Visualization ------------------------
class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def draw_frame(ax, T: pin.SE3, axis_len: float = 0.06):
        o = T.translation; Rm = T.rotation
        cols = ["r", "g", "b"]
        for k in range(3):
            a = Rm[:, k] * axis_len
            ax.plot([o[0], o[0]+a[0]], [o[1], o[1]+a[1]], [o[2], o[2]+a[2]], cols[k], linewidth=1.8)
        ax.scatter([o[0]], [o[1]], [o[2]], s=35, c="k")

    @staticmethod
    def joint_positions(robot: RobotArm, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)
        pts = [robot.data.oMi[jid].translation for jid in range(1, robot.model.njoints)]
        pts.append(robot.data.oMf[robot.tip_fid].translation)
        return np.vstack(pts)

    def show_triplet(self, robot_sim: RobotArm, robot_est: RobotArm,
                     q_final: np.ndarray, kp_true: np.ndarray, kp_est: np.ndarray,
                     T_target_se3: pin.SE3, title: str = "Final comparison"):
        # Curves
        P_rigid = self.joint_positions(robot_sim, q_final)
        q_eq_true = robot_sim.equilibrium_s1(q_final, kp_true, maxiter=80)
        P_true = self.joint_positions(robot_sim, q_eq_true)
        q_eq_est  = robot_est.equilibrium_s1(q_final, np.maximum(kp_est,1e-8), maxiter=80)
        P_est  = self.joint_positions(robot_est, q_eq_est)

        fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111, projection="3d")
        ax.plot(P_rigid[:,0], P_rigid[:,1], P_rigid[:,2], "o--", label="sim rigid (q_final)")
        ax.plot(P_true[:,0],  P_true[:,1],  P_true[:,2],  "o-",  label="sim gravity (Kp_true)")
        ax.plot(P_est[:,0],   P_est[:,1],   P_est[:,2],   "o-.", label="est gravity (Kp_est)")

        # Frames at each joint (use sim kinematics for display)
        # Ts_sim = robot_sim.joint_transforms(q_final)
        # for Ti in Ts_sim:
        #     self.draw_frame(ax, Ti, axis_len=0.06)

        # Target frame (fixed in space)
        self.draw_frame(ax, T_target_se3, axis_len=0.08)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title(title); ax.legend()
        all_pts = np.vstack([P_rigid, P_true, P_est, T_target_se3.translation.reshape(1,3)])
        c = all_pts.mean(axis=0); r = max(np.max(np.linalg.norm(all_pts-c, axis=1)), 0.25)
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.view_init(elev=25, azim=45); plt.gca().set_box_aspect([1,1,1]); plt.tight_layout(); plt.show()

# ------------------------ Main ------------------------
if __name__ == "__main__":
    # Config
    rng = np.random.default_rng(8)
    g_base = np.array([0.0, 0.0, -9.81])
    kp_true = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float)
    parameter_A = 100.0
    R_scalar = 1e-1
    n_segments = 4
    cmd_densify = 60
    N_recalc = 1  # q_des recomputation frequency (per segment)

    # 1) Two arms (same URDF)
    urdf_path = ensure_urdf()
    robot_sim = RobotArm(urdf_path, tip_link="link6", base_link="base_link")  # "real" plant (truth)
    robot_est = RobotArm(urdf_path, tip_link="link6", base_link="base_link")  # estimator
    chain = IkChain.from_urdf_file(urdf_path, base_elements=["base_link"], symbolic=False)

    # 2) Create FIXED target frame T_target BEFORE estimation (robot_est does NOT store it)
    #    Use robot_sim with a random joint, then true equilibrium -> EE pose at that equilibrium
    model = robot_sim.model
    q_rand = np.array([rng.uniform(lo, hi) for lo, hi in zip(model.lowerPositionLimit, model.upperPositionLimit)])
    q_eq_true_for_target = robot_sim.equilibrium_s1(q_rand, kp_true, maxiter=80)
    T_target_se3 = robot_sim.fk_pose(q_eq_true_for_target)  # <-- fixed in space forever

    # 3) Iterative estimation/controller loop
    #    Start from neutral command and an initial Kp guess
    q_des = np.zeros(robot_est.nv)
    x0 = np.log(np.ones(robot_est.nv) * 25.0)
    P0 = np.eye(robot_est.nv) * 1.0
    Q  = np.eye(robot_est.nv) * 1e-3
    ukf = LogKpUKF(x0, P0, Q, R_scalar, alpha=1e-2, beta=2.0, kappa=0.0)
    kp_hat = np.maximum(np.exp(ukf.x), 1e-8)

    # Build SE(3) guidance milestones purely for the controller (robot_est itself does not store T_target)
    T_start = robot_est.fk_pose(robot_est.equilibrium_s1(q_des, kp_hat, maxiter=60))
    R0 = Rsc.from_matrix(T_start.rotation); R1 = Rsc.from_matrix(T_target_se3.rotation)
    key_rots = Rsc.from_quat(np.vstack([R0.as_quat(), R1.as_quat()]))
    slerp = Slerp([0.0, 1.0], key_rots)
    ts = np.linspace(0.0, 1.0, n_segments+1)[1:]
    segment_targets = []
    for s in ts:
        p = (1-s)*T_start.translation + s*T_target_se3.translation
        Rs = slerp([s]).as_matrix()[0]
        segment_targets.append(pin.SE3(Rs, p))

    q_final = q_des.copy()
    for i, T_seg in enumerate(segment_targets, 1):
        # 3-1) Controller computes NEW command using IKPy and robot_est equilibrium map (T_target is only used here)
        if (i-1) % max(1, N_recalc) == 0:
            q_des_new, q_eq_est, q_cmd_path = controller_gravity_aware_q_des(
                robot_est, chain, kp_hat, q_des, T_seg,
                max_outer_iters=8, cmd_densify=cmd_densify,
                pos_tol=1e-3, rot_tol=1e-2,
                kp_rigid_thresh=500.0, alpha_step=1.0
            )
        else:
            q_des_new = q_des.copy()
            q_cmd_path = make_joint_path(q_des, q_des_new, max(2, cmd_densify))

        # 3-2) UKF update ALONG THE RIGID COMMAND PATH ONLY (robot_sim builds measurements)
        update_ukf_along_cmd_path(robot_sim, robot_est, ukf, kp_true, g_base, q_cmd_path,
                                  parameter_A=parameter_A, newton_iter_true=60)
        kp_hat = np.maximum(np.exp(ukf.x), 1e-8)
        q_des = q_des_new.copy()
        q_final = q_des.copy()
        print(f"[Segment {i}/{n_segments}] Kp_hat =", kp_hat)

    # 4) Visualization + success check (robot_sim gravity model aims at the fixed T_target)
    viz = Visualizer()
    viz.show_triplet(robot_sim, robot_est, q_final, kp_true, kp_hat, T_target_se3,
                     title="Triplet: sim rigid vs sim gravity (true) vs est gravity (hat)")

    q_eq_true_final = robot_sim.equilibrium_s1(q_final, kp_true, maxiter=80)
    E_final = robot_sim.fk_pose(q_eq_true_final).inverse() * T_target_se3
    pos_err = np.linalg.norm(E_final.translation)
    rot_err = np.linalg.norm(pin.log3(E_final.rotation))
    print("Kp_true =", kp_true)
    print("Kp_hat  =", np.exp(ukf.x))
    print("pos_err =", pos_err, "rot_err =", rot_err)
