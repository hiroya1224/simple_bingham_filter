# RIK→θ_ref path, GaIK(逆静力学)→θ_cmd path, analytic Weird EKF on log Kp
# - Two arms (robot_sim: truth, robot_est: estimation), same URDF
# - Target SE(3) is fixed beforehand by robot_sim (never moved)
# - IK: always IKPy (rigid)
# - S^1 equilibrium solver: mandatory homotopy (lambda=linspace(1,0,10), k_stiffness=100), no early break
# - All linear algebra uses pinv (no solve/inv)

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
  <joint name="joint6" type="revololute">
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
    T = np.eye(4); T[:3,:3] = M.rotation; T[:3,3] = M.translation
    return T

# ------------------------ Bingham helpers (NO normalization) ------------------------
def Q_from_quat_wxyz(z: np.ndarray) -> np.ndarray:
    w, x, y, zc = z
    return np.array([
        [-x,   -y,   -zc],
        [ w,   -zc,   y ],
        [ zc,   w,   -x ],
        [-y,    x,    w ]
    ], dtype=float)

def simple_bingham_unit(before_vec3: np.ndarray, after_vec3: np.ndarray, parameter: float = 100.0) -> np.ndarray:
    b = np.asarray(before_vec3, dtype=float); a = np.asarray(after_vec3, dtype=float)
    bn = b / (np.linalg.norm(b) + 1e-12); an = a / (np.linalg.norm(a) + 1e-12)
    vq = np.array([0.0, bn[0], bn[1], bn[2]], dtype=float)
    xq = np.array([0.0, an[0], an[1], an[2]], dtype=float)
    def Lmat(q):
        a,b,c,d = q
        return np.array([[a,-b,-c,-d],[b,a,-d,c],[c,d,a,-b],[d,-c,b,a]])
    def Rmat(q):
        w,x,y,z = q
        return np.array([[w,-x,-y,-z],[x,w,z,-y],[y,-z,w,x],[z,y,-x,w]])
    P = Lmat(xq) - Rmat(vq)
    A0 = -0.25*(P.T @ P)  # semi-negative definite
    return float(parameter)*A0

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
        self.eq_path_last: List[np.ndarray] = []

    # --- mandatory-homotopy S^1-LM equilibrium (no early break) ---
    def equilibrium_s1(self, theta_cmd: np.ndarray, kp_vec: np.ndarray,
                       maxiter: int = 80, eps: float = 1e-10,
                       theta_init: Optional[np.ndarray] = None,
                       lm_mu0: float = 1e-1, lm_mu_max: float = 1e8,
                       step_clip: float = 0.30, monotonic: bool = True,
                       k_stiffness: float = 100.0,
                       lambdas: Optional[np.ndarray] = None) -> np.ndarray:
        if lambdas is None:
            lambdas = np.linspace(1.0, 0.0, 10)
        n = self.nv
        K = np.diag(kp_vec)
        theta0 = theta_init.copy() if theta_init is not None else theta_cmd.copy()

        def cs_from_theta(th):
            return np.cos(th), np.sin(th)
        def rotate_cs(c, s, dth):
            cd, sd = np.cos(dth), np.sin(dth)
            c2 = c*cd - s*sd
            s2 = s*cd + c*sd
            nrm = np.sqrt(c2*c2 + s2*s2)
            return c2/nrm, s2/nrm
        def theta_from_cs(c, s):
            return np.arctan2(s, c)

        it_per_stage = max(1, int(np.ceil(maxiter / max(1, len(lambdas)))))
        self.eq_path_last = []
        c, s = cs_from_theta(theta0)

        for lam in lambdas:
            K_eff = K + (lam * k_stiffness) * np.eye(n)
            mu = float(lm_mu0)
            for _ in range(it_per_stage):
                th = theta_from_cs(c, s)
                tau_g = pin.computeGeneralizedGravity(self.model, self.data, th)
                F = tau_g + K_eff @ (th - theta_cmd)

                dG = pin.computeGeneralizedGravityDerivatives(self.model, self.data, th)
                Jq = dG + K_eff

                JT = Jq.T
                A = JT @ Jq + mu * np.eye(n)
                b = - JT @ F
                # pinv-based LM step
                dq = np.linalg.pinv(A, rcond=1e-12) @ b
                dq = np.clip(dq, -step_clip, step_clip)

                c_try, s_try = rotate_cs(c, s, dq)
                th_try = theta_from_cs(c_try, s_try)
                F_try = pin.computeGeneralizedGravity(self.model, self.data, th_try) + K_eff @ (th_try - theta_cmd)

                if (not monotonic) or (np.linalg.norm(F_try) <= np.linalg.norm(F)):
                    c, s = c_try, s_try
                    mu = max(lm_mu0, mu * 0.33)
                else:
                    mu = min(lm_mu_max, mu * 3.0)

            self.eq_path_last.append(theta_from_cs(c, s).copy())

        theta_fin = theta_from_cs(c, s)
        return theta_fin

    # kinematics helpers
    def ee_rotation_in_base(self, theta: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, theta); pin.updateFramePlacements(self.model, self.data)
        R_wb = self.data.oMf[self.base_fid].rotation
        R_we = self.data.oMf[self.tip_fid].rotation
        return R_wb.T @ R_we

    def ee_quaternion_wxyz_base(self, theta: np.ndarray) -> np.ndarray:
        R_be = self.ee_rotation_in_base(theta)
        q_xyzw = Rsc.from_matrix(R_be).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        return q_wxyz / (np.linalg.norm(q_wxyz) + 1e-18)

    def gravity_dir_in_ee(self, theta: np.ndarray, g_base: np.ndarray) -> np.ndarray:
        R_be = self.ee_rotation_in_base(theta)
        gb = g_base / (np.linalg.norm(g_base) + 1e-12)
        gee = R_be @ gb
        return gee / (np.linalg.norm(gee) + 1e-12)

    def fk_pose(self, theta: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, theta); pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tip_fid]

    def frame_angular_jacobian_world(self, theta: np.ndarray) -> np.ndarray:
        pin.computeJointJacobians(self.model, self.data, theta)
        pin.updateFramePlacements(self.model, self.data)
        J6 = pin.computeFrameJacobian(self.model, self.data, theta, self.tip_fid, pin.ReferenceFrame.WORLD)
        return J6[3:6, :]

    def joint_transforms(self, theta: np.ndarray) -> List[pin.SE3]:
        pin.forwardKinematics(self.model, self.data, theta); pin.updateFramePlacements(self.model, self.data)
        Ts = [self.data.oMi[jid] for jid in range(1, self.model.njoints)]
        Ts.append(self.data.oMf[self.tip_fid])
        return Ts

# ------------------------ IK (RIK) ------------------------
def rik_path_from_targets(chain: IkChain, T_list: List[pin.SE3],
                          theta_seed: np.ndarray, max_iter: int = 1200) -> np.ndarray:
    thetas = []
    theta_prev = theta_seed.copy()
    for T in T_list:
        T_h = se3_to_homog(T)
        q0_full = np.zeros(len(chain.links))
        q0_full[1:1+theta_prev.size] = theta_prev
        sol_full = chain.inverse_kinematics_frame(
            T_h, initial_position=q0_full, max_iter=max_iter, orientation_mode="all"
        )
        theta_ref = np.array(sol_full[1:1+theta_prev.size])
        thetas.append(theta_ref)
        theta_prev = theta_ref
    return np.vstack(thetas)

# ------------------------ GaIK (inverse statics one-shot) ------------------------
def theta_cmd_from_theta_ref(robot: RobotArm, theta_ref: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
    tau_g = pin.computeGeneralizedGravity(robot.model, robot.data, theta_ref)
    return theta_ref + tau_g / np.maximum(kp_vec, 1e-12)

# ------------------------ A_t from robot_sim (truth) ------------------------
def build_A_from_true(robot_sim: RobotArm, theta_cmd: np.ndarray, kp_true: np.ndarray, g_base: np.ndarray,
                      parameter_A: float = 100.0, newton_iter_true: int = 60,
                      theta_ws_true: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    theta_equil_true = robot_sim.equilibrium_s1(theta_cmd, kp_true, maxiter=newton_iter_true, theta_init=theta_ws_true)
    gee_true = robot_sim.gravity_dir_in_ee(theta_equil_true, g_base)
    A_t = simple_bingham_unit(g_base, gee_true, parameter=parameter_A)
    return A_t, theta_equil_true

# ------------------------ Analytic Weird EKF (pinv-only) ------------------------
class AnalyticWeirdEKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, eps_def: float = 1e-6):
        self.x = x0.copy()   # mean of log Kp
        self.P = P0.copy()
        self.Q = Q.copy()
        self.eps_def = float(eps_def)

    def predict(self):
        self.P = self.P + self.Q

    def _grad_hess_analytic(self, x0: np.ndarray, theta_cmd: np.ndarray, A_t: np.ndarray,
                            robot_est: RobotArm, theta_init: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        n = x0.size
        k = np.exp(x0)
        K = np.diag(k)
        # equilibrium at current mean, warm-start with predicted or theta_ref
        theta_eq = robot_est.equilibrium_s1(theta_cmd, k, maxiter=80, theta_init=theta_init)

        dG = pin.computeGeneralizedGravityDerivatives(robot_est.model, robot_est.data, theta_eq)  # (n x n)
        J_q = dG + K
        J_x = np.diag(k * (theta_eq - theta_cmd))
        J_w = robot_est.frame_angular_jacobian_world(theta_eq)  # (3 x n)
        z = robot_est.ee_quaternion_wxyz_base(theta_eq)         # (4,)
        Qz = Q_from_quat_wxyz(z)                                # (4 x 3)

        # gradient g = - J_x^T J_q^{-T} J_w^T Q^T A z
        v = Qz.T @ (A_t @ z)                   # (3,)
        u = J_w.T @ v                          # (n,)
        y = np.linalg.pinv(J_q.T, rcond=1e-12) @ u
        g = - (J_x.T @ y)                      # (n,)

        # Hessian H0 = 1/2 * M^T A M,  M = Q J_w J_q^{-1} J_x
        X = np.linalg.pinv(J_q, rcond=1e-12) @ J_x   # (n x n)
        M = Qz @ (J_w @ X)                            # (4 x n)
        H0 = 0.5 * (M.T @ (A_t @ M))
        MtM = M.T @ M

        # Bingham invariance: shift by c*I_4 ⇒ H = H0 + 1/2 c MtM, choose c ≤ 0 s.t. -H ≻ 0
        H0s = 0.5*(H0 + H0.T)
        wH = np.linalg.eigvalsh(H0s)
        lam_max_H = float(np.max(wH)) if wH.size > 0 else 0.0
        if lam_max_H <= -self.eps_def:
            H = H0s
        else:
            Bs = 0.5*(MtM + MtM.T)
            wB = np.linalg.eigvalsh(Bs)
            lam_max_B = float(np.max(wB)) if wB.size > 0 else 0.0
            if lam_max_B <= 1e-12:
                H = H0s - (lam_max_H + self.eps_def) * np.eye(n)
            else:
                c = - 2.0 * (lam_max_H + self.eps_def) / lam_max_B
                H = H0s + 0.5 * c * MtM
        return g, H

    def update_with_L_quadratic(self, theta_cmd: np.ndarray, A_t: np.ndarray,
                                robot_est: RobotArm, theta_init_eq_pred: Optional[np.ndarray]):
        self.predict()
        g, H = self._grad_hess_analytic(self.x, theta_cmd, A_t, robot_est, theta_init_eq_pred)
        Sinv = -H  # precision of pseudo-observation
        # ensure PD by minimal diagonal boost if needed (pinv-friendly)
        w = np.linalg.eigvalsh(0.5*(Sinv+Sinv.T))
        lam_min = float(np.min(w))
        if lam_min <= self.eps_def:
            Sinv = Sinv + ((self.eps_def - lam_min) + 1e-12) * np.eye(Sinv.shape[0])
        S = np.linalg.pinv(Sinv, rcond=1e-12)
        m = self.x + S @ g

        Pinv = np.linalg.pinv(self.P, rcond=1e-12)
        J_post = Pinv + Sinv
        P_post = np.linalg.pinv(J_post, rcond=1e-12)
        h_post = Pinv @ self.x + Sinv @ m
        x_post = P_post @ h_post

        self.P = 0.5*(P_post + P_post.T)
        self.x = x_post

# ------------------------ Visualization ------------------------
class Visualizer:
    @staticmethod
    def draw_frame(ax, T: pin.SE3, axis_len: float = 0.06):
        o = T.translation; Rm = T.rotation
        cols = ["r", "g", "b"]
        for k in range(3):
            a = Rm[:, k] * axis_len
            ax.plot([o[0], o[0]+a[0]], [o[1], o[1]+a[1]], [o[2], o[2]+a[2]], cols[k], linewidth=1.8)
        ax.scatter([o[0]], [o[1]], [o[2]], s=35, c="k")

    @staticmethod
    def joint_positions(robot: RobotArm, theta: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(robot.model, robot.data, theta)
        pin.updateFramePlacements(robot.model, robot.data)
        pts = [robot.data.oMi[jid].translation for jid in range(1, robot.model.njoints)]
        pts.append(robot.data.oMf[robot.tip_fid].translation)
        return np.vstack(pts)

    def show_triplet(self, robot_sim: RobotArm, robot_est: RobotArm,
                     theta_cmd_final: np.ndarray, kp_true: np.ndarray, kp_est: np.ndarray,
                     T_target_se3: pin.SE3, title: str = "Final comparison"):
        P_rigid = self.joint_positions(robot_sim, theta_cmd_final)
        theta_equil_true = robot_sim.equilibrium_s1(theta_cmd_final, kp_true, maxiter=80, theta_init=theta_cmd_final)
        P_true = self.joint_positions(robot_sim, theta_equil_true)
        theta_equil_est  = robot_est.equilibrium_s1(theta_cmd_final, np.maximum(kp_est,1e-8),
                                                    maxiter=80, theta_init=theta_cmd_final)
        P_est  = self.joint_positions(robot_est, theta_equil_est)

        fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111, projection="3d")
        ax.plot(P_rigid[:,0], P_rigid[:,1], P_rigid[:,2], "o--", label="sim rigid (theta_cmd)")
        ax.plot(P_true[:,0],  P_true[:,1],  P_true[:,2],  "o-",  label="sim gravity (Kp_true)")
        ax.plot(P_est[:,0],   P_est[:,1],   P_est[:,2],   "o-.", label="est gravity (Kp_est)")

        # Ts_sim = robot_sim.joint_transforms(theta_cmd_final)
        # for Ti in Ts_sim:
        #     self.draw_frame(ax, Ti, axis_len=0.06)
        self.draw_frame(ax, T_target_se3, axis_len=0.08)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title(title); ax.legend()
        all_pts = np.vstack([P_rigid, P_true, P_est, T_target_se3.translation.reshape(1,3)])
        c = all_pts.mean(axis=0); r = max(np.max(np.linalg.norm(all_pts-c, axis=1)), 0.25)
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.view_init(elev=25, azim=45); plt.gca().set_box_aspect([1,1,1]); plt.tight_layout(); plt.show()

# ------------------------ Main ------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(8)
    g_base = np.array([0.0, 0.0, -9.81])
    kp_true = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float)
    parameter_A = 100.0
    n_waypoints = 60

    # Two arms + IKPy chain
    urdf_path = ensure_urdf()
    robot_sim = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
    robot_est = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
    chain = IkChain.from_urdf_file(urdf_path, base_elements=["base_link"], symbolic=False)

    # Fixed target (built before estimation on robot_sim)
    theta_rand = np.array([rng.uniform(lo, hi)
                           for lo, hi in zip(robot_sim.model.lowerPositionLimit,
                                             robot_sim.model.upperPositionLimit)])
    theta_equil_true_for_target = robot_sim.equilibrium_s1(theta_rand, kp_true, maxiter=80, theta_init=theta_rand)
    T_target_se3 = robot_sim.fk_pose(theta_equil_true_for_target)

    # Build SE(3) waypoint list from start pose to target
    theta_seed = np.zeros(robot_est.nv)
    theta_equil_seed = robot_est.equilibrium_s1(theta_seed, np.ones(robot_est.nv)*25.0, maxiter=60, theta_init=theta_seed)
    T_start = robot_est.fk_pose(theta_equil_seed)
    R0 = Rsc.from_matrix(T_start.rotation); R1 = Rsc.from_matrix(T_target_se3.rotation)
    slerp = Slerp([0.0, 1.0], Rsc.from_quat(np.vstack([R0.as_quat(), R1.as_quat()])))
    T_list = []
    for s in np.linspace(0.0, 1.0, n_waypoints):
        p = (1-s)*T_start.translation + s*T_target_se3.translation
        Rm = slerp([s]).as_matrix()[0]
        T_list.append(pin.SE3(Rm, p))

    # RIK: θ_ref path
    theta_ref_path = rik_path_from_targets(chain, T_list, theta_seed, max_iter=1200)

    # Weird EKF init (log Kp)
    x0 = np.log(np.ones(robot_est.nv) * 25.0)
    P0 = np.eye(robot_est.nv) * 1.0
    Q  = np.eye(robot_est.nv) * 1e-3
    wekf = AnalyticWeirdEKF(x0, P0, Q, eps_def=1e-6)

    # Loop over θ_ref path → θ_cmd → A_t → EKF update
    theta_ws_true = None
    theta_ws_est_pred = None
    theta_cmd_final = None
    for k in range(theta_ref_path.shape[0]):
        theta_ref_k = theta_ref_path[k]
        kp_hat = np.exp(wekf.x)

        # GaIK (inverse statics): θ_cmd from θ_ref
        theta_cmd_k = theta_cmd_from_theta_ref(robot_est, theta_ref_k, kp_hat)
        theta_cmd_final = theta_cmd_k

        # Build observation A_t on robot_sim using θ_cmd_k
        A_t, theta_equil_true_k = build_A_from_true(robot_sim, theta_cmd_k, kp_true, g_base,
                                                    parameter_A=parameter_A, newton_iter_true=60,
                                                    theta_ws_true=theta_ws_true)
        theta_ws_true = theta_equil_true_k

        # Predicted equilibrium on robot_est (warm-start with θ_ref_k)
        theta_equil_pred_k = robot_est.equilibrium_s1(theta_cmd_k, kp_hat, maxiter=80, theta_init=theta_ref_k)
        theta_ws_est_pred = theta_equil_pred_k

        # Weird EKF update (pinv-only), using θ_cmd and warm-start
        wekf.update_with_L_quadratic(theta_cmd_k, A_t, robot_est, theta_init_eq_pred=theta_equil_pred_k)

        # (optional) print progress
        if (k+1) % max(1, n_waypoints//4) == 0:
            print(f"[{k+1}/{n_waypoints}] Kp_hat =", np.exp(wekf.x))

    # Visualization & report
    viz = Visualizer()
    viz.show_triplet(robot_sim, robot_est, theta_cmd_final, kp_true, np.maximum(np.exp(wekf.x),1e-8),
                     T_target_se3, title="Triplet: rigid vs true-gravity vs est-gravity (RIK→GaIK)")

    theta_equil_true_final = robot_sim.equilibrium_s1(theta_cmd_final, kp_true, maxiter=80, theta_init=theta_cmd_final)
    E_final = robot_sim.fk_pose(theta_equil_true_final).inverse() * T_target_se3
    pos_err = np.linalg.norm(E_final.translation)
    rot_err = np.linalg.norm(pin.log3(E_final.rotation))
    print("Kp_true =", kp_true)
    print("Kp_hat  =", np.exp(wekf.x))
    print("pos_err =", pos_err, "rot_err =", rot_err)
