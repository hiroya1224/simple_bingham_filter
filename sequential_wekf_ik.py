# Gravity-aware reaching with "analytic weird EKF" on Kp (closed-form gradient/Hessian)
# - Target frame fixed beforehand by robot_sim; robot_est does not store it internally.
# - IK: always IKPy (rigid). Command path only is fed to the filter.
# - Log-likelihood: L(x)=z(x)^T A z(x), with z = unit quaternion (wxyz) of EE in base.
# - Gradient/Hessian (Gauss–Newton) are analytic:
#     dqeq/dx = - J_q^{-1} J_x,  J_x=diag(k*(q_eq-q_cmd)),  J_q = dG + K
#     z'(x) = (1/2) Q(z) J_w dqeq/dx
#     g = - J_x^T J_q^{-T} J_w^T Q(z)^T A z
#     H0 = 1/2 * M^T A M,  M = Q(z) J_w J_q^{-1} J_x
#   Use Bingham invariance: H = H0 + 1/2 * c * (M^T M), choose c<=0 to ensure H is negative-definite.

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
        for i in range(1, 6+1)
    ]))
        with open(urdf_path, "w") as f:
            f.write(txt)
    return urdf_path

def se3_to_homog(M: pin.SE3) -> np.ndarray:
    T = np.eye(4); T[:3,:3] = M.rotation; T[:3,3] = M.translation
    return T

def make_joint_path(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    a = np.linspace(0.0, 1.0, n)
    return (1-a)[:,None]*q0[None,:] + a[:,None]*q1[None,:]

# ------------------------ Bingham helpers ------------------------
def Q_from_quat_wxyz(z: np.ndarray) -> np.ndarray:
    # z = [w,x,y,z], Q in R^{4x3}
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
    # Quaternion of pure vectors (0,v)
    vq = np.array([0.0, bn[0], bn[1], bn[2]], dtype=float)
    xq = np.array([0.0, an[0], an[1], an[2]], dtype=float)
    # P = L(xq) - R(vq)
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

    def equilibrium_s1(self, q_des: np.ndarray, kp_vec: np.ndarray,
                    maxiter: int = 80, eps: float = 1e-10,
                    q_init: Optional[np.ndarray] = None,
                    # LM / trust-region
                    lm_mu0: float = 1e-1, lm_mu_max: float = 1e8,
                    step_clip: float = 0.30,        # [rad] per iter per joint (S^1)
                    monotonic: bool = True,
                    # homotopy (always on)
                    k_stiffness: float = 1000.0,
                    lambdas: Optional[np.ndarray] = None
                    ) -> np.ndarray:
        """
        Solve g(q) + K(q - q_des) = 0 on S^1 with mandatory homotopy:
        K_eff = K + lambda * k_stiffness * I,  lambda goes 1 -> 0 (default 10 steps).
        For each lambda stage:
        - run fixed number of LM iterations on S^1 (NO early break)
        - warm start at previous stage's equilibrium
        - record the stage equilibrium to self.eq_path_last
        Return the final equilibrium at lambda=0.
        Caller SHOULD pass q_init = q_cmd (rigid command).
        """
        # init state on S^1 (start from q_init if given; else from q_des)
        q0 = q_init.copy() if q_init is not None else q_des.copy()
        n = self.nv
        K = np.diag(kp_vec)
        if lambdas is None:
            lambdas = np.linspace(1.0, 0.0, 100)  # mandatory schedule

        # helpers: S^1 representation and update
        def cs_from_q(q):
            return np.cos(q), np.sin(q)
        def rotate_cs(c, s, dq):
            cd, sd = np.cos(dq), np.sin(dq)
            c2 = c*cd - s*sd
            s2 = s*cd + c*sd
            nrm = np.sqrt(c2*c2 + s2*s2)
            return c2/nrm, s2/nrm
        def q_from_cs(c, s):
            return np.arctan2(s, c)

        # number of inner iterations per homotopy stage (no early break)
        it_per_stage = max(1, int(np.ceil(maxiter / max(1, len(lambdas)))))

        # storage for stage-wise equilibria (for later inspection)
        self.eq_path_last = []

        # initialize at start
        c, s = cs_from_q(q0)

        # homotopy loop: always run full schedule, no break
        for lam in lambdas:
            K_eff = K + (lam * k_stiffness) * np.eye(n)
            mu = float(lm_mu0)

            for _ in range(it_per_stage):
                q = q_from_cs(c, s)

                # residual and Jacobian in q-space (with K_eff)
                tau_g = pin.computeGeneralizedGravity(self.model, self.data, q)
                F = tau_g + K_eff @ (q - q_des)

                dG = pin.computeGeneralizedGravityDerivatives(self.model, self.data, q)
                Jq = dG + K_eff  # dF/dq with effective stiffness

                # LM step for minimizing 1/2 ||F||^2
                JT = Jq.T
                A = JT @ Jq + mu * np.eye(n)
                b = - JT @ F
                try:
                    dq = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    dq = np.linalg.lstsq(A, b, rcond=1e-12)[0]

                # trust region: clip per-joint step on S^1
                dq = np.clip(dq, -step_clip, step_clip)

                # trial rotation & accept/reject (monotone option)
                c_try, s_try = rotate_cs(c, s, dq)
                q_try = q_from_cs(c_try, s_try)
                F_try = pin.computeGeneralizedGravity(self.model, self.data, q_try) + K_eff @ (q_try - q_des)

                if (not monotonic) or (np.linalg.norm(F_try) <= np.linalg.norm(F)):
                    c, s = c_try, s_try
                    mu = max(lm_mu0, mu * 0.33)
                else:
                    mu = min(lm_mu_max, mu * 3.0)

            # record equilibrium at this lambda stage (after fixed iterations)
            self.eq_path_last.append(q_from_cs(c, s).copy())

        # final equilibrium (lambda=0) is the last state
        q_fin = q_from_cs(c, s)
        return q_fin
  
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

    def frame_angular_jacobian_world(self, q: np.ndarray) -> np.ndarray:
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J6 = pin.computeFrameJacobian(self.model, self.data, q, self.tip_fid, pin.ReferenceFrame.WORLD)
        # Pinocchio returns 6xn: top=linear, bottom=angular
        return J6[3:6, :]

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

def controller_one_step(robot_est: RobotArm, chain: IkChain, kp_vec: np.ndarray,
                        q_des: np.ndarray, T_target_se3: pin.SE3,
                        densify: int = 30, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    T_target = se3_to_homog(T_target_se3)
    q_rigid = ikpy_inverse(chain, T_target, q_des, max_iter=800)
    q_new = (1.0 - alpha) * q_des + alpha * q_rigid
    q_cmd_path = make_joint_path(q_des, q_new, max(2, densify))
    return q_new, q_cmd_path

# ------------------------ A from robot_sim (truth) ------------------------
def build_A_from_true(robot_sim: RobotArm, q_cmd: np.ndarray, kp_true: np.ndarray, g_base: np.ndarray,
                      parameter_A: float = 100.0, newton_iter_true: int = 60,
                      qeq_ws_true: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    q_eq_true = robot_sim.equilibrium_s1(q_cmd, kp_true, maxiter=newton_iter_true, q_init=qeq_ws_true)
    gee_true = robot_sim.gravity_dir_in_ee(q_eq_true, g_base)
    A_t = simple_bingham_unit(g_base, gee_true, parameter=parameter_A)
    return A_t, q_eq_true

# ------------------------ Analytic Weird EKF ------------------------
class AnalyticWeirdEKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray,
                 eps_def: float = 1e-6):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.eps_def = float(eps_def)

    def predict(self):
        self.P = self.P + self.Q

    @staticmethod
    def _solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Solve A X = B stably
        try:
            return np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, B, rcond=1e-12)[0]

    def _grad_hess_analytic(self, x0: np.ndarray, q_cmd: np.ndarray, A_t: np.ndarray,
                            robot_est: RobotArm) -> Tuple[np.ndarray, np.ndarray]:
        n = x0.size
        k = np.exp(x0)
        K = np.diag(k)
        # equilibrium at x0
        q_eq = robot_est.equilibrium_s1(q_cmd, k, maxiter=60)
        # needed Jacobians
        dG = pin.computeGeneralizedGravityDerivatives(robot_est.model, robot_est.data, q_eq)  # (n x n)
        J_q = dG + K  # (n x n)
        J_x = np.diag(k * (q_eq - q_cmd))  # (n x n)
        J_w = robot_est.frame_angular_jacobian_world(q_eq)  # (3 x n)
        z = robot_est.ee_quaternion_wxyz_base(q_eq)  # (4,)
        Qz = Q_from_quat_wxyz(z)  # (4 x 3)

        # Build pieces
        # Gradient: g = - J_x^T J_q^{-T} J_w^T Q^T A z
        v = Qz.T @ (A_t @ z)            # (3,)
        u = J_w.T @ v                   # (n,)
        y = self._solve(J_q.T, u)       # (n,) solves J_q^T y = u
        g = - (J_x.T @ y)               # (n,)

        # Hessian (Gauss–Newton): H0 = 1/2 * M^T A M, M = Q J_w J_q^{-1} J_x  (size 4 x n)
        X = self._solve(J_q, J_x)       # (n x n), J_q X = J_x
        M = Qz @ (J_w @ X)              # (4 x n)
        H0 = 0.5 * (M.T @ (A_t @ M))    # (n x n), symmetric up to num. error
        # Stabilize with Bingham invariance: add 1/2 * c * (M^T M)
        MtM = M.T @ M

        # choose minimal negative c so that H=H0 + 1/2*c*MtM has lambda_max <= -eps_def
        # if already negative enough, c=0
        # estimate spectral radii
        wH = np.linalg.eigvalsh(0.5*(H0+H0.T))
        lam_max_H = float(np.max(wH)) if wH.size > 0 else 0.0
        if lam_max_H <= -self.eps_def:
            c = 0.0
            H = 0.5*(H0 + H0.T)
        else:
            wB = np.linalg.eigvalsh(0.5*(MtM+MtM.T))
            lam_max_B = float(np.max(wB)) if wB.size > 0 else 0.0
            if lam_max_B <= 1e-12:
                # fallback: isotropic shift on variables
                alpha = lam_max_H + self.eps_def
                H = 0.5*(H0 + H0.T) - alpha * np.eye(n)
                c = None  # not used
            else:
                c = - 2.0 * (lam_max_H + self.eps_def) / lam_max_B  # negative
                H = 0.5*(H0 + H0.T) + 0.5 * c * MtM  # symmetric by construction

        return g, H

    def update_with_L_quadratic(self, q_cmd: np.ndarray, A_t: np.ndarray,
                                robot_est: RobotArm):
        # Prior
        self.predict()
        # Analytic gradient/Hessian at current mean
        g, H = self._grad_hess_analytic(self.x, q_cmd, A_t, robot_est)
        # Build "measurement" Gaussian from quadratic log-likelihood:
        #   log p(y|x) ~ const + g^T (x - x0) + 1/2 (x - x0)^T H (x - x0)
        # => In information form at x0:
        #   S^{-1} = -H    (must be PD),    m = x0 + (-H)^{-1} g
        Sinv = -H
        Sinv = 0.5*(Sinv+Sinv.T)
        # Safeguard PD
        w = np.linalg.eigvalsh(Sinv)
        lam_min = float(np.min(w))
        
        if lam_min <= self.eps_def:
            boost = (self.eps_def - lam_min) + 1e-3
            Sinv = Sinv + boost * np.eye(Sinv.shape[0])

        # m = self.x + np.linalg.solve(Sinv, g)
        m = self.x + np.dot(np.linalg.pinv(Sinv), g)

        # Fuse prior N(x|x,P) with "measurement" N(x|m,S) via information addition
        Pinv = np.linalg.pinv(self.P)
        J_post = Pinv + Sinv
        h_post = Pinv @ self.x + Sinv @ m
        self.P = np.linalg.pinv(J_post)
        self.x = self.P @ h_post

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
        P_rigid = self.joint_positions(robot_sim, q_final)
        q_eq_true = robot_sim.equilibrium_s1(q_final, kp_true, maxiter=80)
        P_true = self.joint_positions(robot_sim, q_eq_true)
        q_eq_est  = robot_est.equilibrium_s1(q_final, np.maximum(kp_est,1e-8), maxiter=80)
        P_est  = self.joint_positions(robot_est, q_eq_est)

        fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111, projection="3d")
        ax.plot(P_rigid[:,0], P_rigid[:,1], P_rigid[:,2], "o--", label="sim rigid (q_final)")
        ax.plot(P_true[:,0],  P_true[:,1],  P_true[:,2],  "o-",  label="sim gravity (Kp_true)")
        ax.plot(P_est[:,0],   P_est[:,1],   P_est[:,2],   "o-.", label="est gravity (Kp_est)")

        # Ts_sim = robot_sim.joint_transforms(q_final)
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
    # Config
    rng = np.random.default_rng(3)
    g_base = np.array([0.0, 0.0, -9.81])
    kp_true = np.array([4.0, 5.0, 4.0, 9.0, 5.0, 5.0], dtype=float) * 3
    parameter_A = 100.0
    n_segments = 2
    densify = 200
    eps_def = 1e-3

    # 1) Two arms
    urdf_path = ensure_urdf()
    robot_sim = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
    robot_est = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
    chain = IkChain.from_urdf_file(urdf_path, base_elements=["base_link"], symbolic=False)

    # 2) Fixed target built BEFORE estimation on robot_sim
    q_rand = np.array([rng.uniform(lo, hi)
                       for lo, hi in zip(robot_sim.model.lowerPositionLimit,
                                         robot_sim.model.upperPositionLimit)])
    q_eq_true_for_target = robot_sim.equilibrium_s1(q_rand, kp_true, maxiter=100)
    T_target_se3 = robot_sim.fk_pose(q_eq_true_for_target)

    # 3) Init filter (log Kp prior)
    q_des = np.zeros(robot_est.nv)
    x0 = np.log(np.ones(robot_est.nv) * 10.0)
    P0 = np.eye(robot_est.nv) * 1.0
    Q  = np.eye(robot_est.nv) * 1e-3
    wekf = AnalyticWeirdEKF(x0, P0, Q, eps_def=eps_def)

    # Build coarse SE(3) milestones for the controller
    T_start = robot_est.fk_pose(robot_est.equilibrium_s1(q_des, np.exp(x0), maxiter=60))
    R0 = Rsc.from_matrix(T_start.rotation); R1 = Rsc.from_matrix(T_target_se3.rotation)
    key_rots = Rsc.from_quat(np.vstack([R0.as_quat(), R1.as_quat()]))
    slerp = Slerp([0.0, 1.0], key_rots)
    ts = np.linspace(0.0, 1.0, n_segments+1)[1:]
    # ts = np.append(1 - np.exp(-5 * ts), [1.0])
    print(ts)
    segment_targets = []
    for s in ts:
        p = (1-s)*T_start.translation + s*T_target_se3.translation
        Rs = slerp([s]).as_matrix()[0]
        segment_targets.append(pin.SE3(Rs, p))

    q_final = q_des.copy()
    qeq_ws_true = None

    for i, T_seg in enumerate(segment_targets, 1):
        # Controller: one IKPy step toward the milestone
        q_des_new, q_cmd_path = controller_one_step(robot_est, chain, np.exp(wekf.x), q_des, T_seg,
                                                    densify=densify, alpha=1.0)
        # Along the RIGID command path, build A_t from robot_sim (truth) and update analytic weird-EKF
        for qc in q_cmd_path:
            A_t, q_eq_true = build_A_from_true(robot_sim, qc, kp_true, g_base,
                                               parameter_A=parameter_A, newton_iter_true=60,
                                               qeq_ws_true=qeq_ws_true)
            qeq_ws_true = q_eq_true
            wekf.update_with_L_quadratic(qc, A_t, robot_est)

        q_des = q_des_new.copy()
        q_final = q_des.copy()
        print(f"[Segment {i}/{n_segments}] Kp_hat =", np.exp(wekf.x))

    # 4) Visualization and report
    viz = Visualizer()
    viz.show_triplet(robot_sim, robot_est, q_final, kp_true, np.maximum(np.exp(wekf.x),1e-8),
                     T_target_se3, title="Triplet: sim rigid vs sim gravity (true) vs est gravity (analytic)")

    q_eq_true_final = robot_sim.equilibrium_s1(q_final, kp_true, maxiter=80)
    E_final = robot_sim.fk_pose(q_eq_true_final).inverse() * T_target_se3
    pos_err = np.linalg.norm(E_final.translation)
    rot_err = np.linalg.norm(pin.log3(E_final.rotation))
    print("Kp_true =", kp_true)
    print("Kp_hat  =", np.exp(wekf.x))
    print("pos_err =", pos_err, "rot_err =", rot_err)
