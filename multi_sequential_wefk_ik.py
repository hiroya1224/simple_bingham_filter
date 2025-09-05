# Multi-frame gravity-direction likelihood for Weird EKF (sum over selected link frames)
# - RIK: one-shot IKPy to target, then linear interpolation from theta_init to theta_ref_goal
# - GaIK: inverse statics one-shot  theta_cmd = theta_ref + K^{-1} tau_g(theta_ref)
# - Equilibrium solver: energy-minimizing S^1 Newton with mandatory homotopy (lambda linspace 1->0, k_stiffness=100)
# - Observations: for frames {f}, A_f built from robot_sim (truth) gravity direction at equilibrium(theta_cmd, Kp_true)
# - Likelihood: L(x) = sum_f z_f(theta_eq(x; theta_cmd))^T A_f z_f(...)
# - Analytic gradient/Hessian summed over frames; Bingham shift to make -H positive-definite
# - All linear algebra uses pinv (no solve/inv). IK must be IKPy. Variable names are ASCII only.

import os
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
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
        self.total_mass = sum(inert.mass for inert in self.model.inertias)
        self.eq_path_last: List[np.ndarray] = []

    # ---- Energy-minimizing S^1 Newton with mandatory homotopy (pinv only) ----
    def equilibrium_s1(self, theta_cmd: np.ndarray, kp_vec: np.ndarray,
                       maxiter: int = 80, theta_init: Optional[np.ndarray] = None,
                       k_stiffness: float = 100.0,
                       lambdas: Optional[np.ndarray] = None,
                       newton_mu0: float = 1e-2, newton_mu_max: float = 1e6,
                       step_clip: float = 0.35, backtrack_max: int = 6,
                       backtrack_shrink: float = 0.5,
                       ensure_spd: bool = True, spd_boost: float = 1e-6) -> np.ndarray:
        if lambdas is None:
            lambdas = np.linspace(1.0, 0.0, 10)
        n = self.nv
        K = np.diag(kp_vec)
        theta0 = theta_init.copy() if theta_init is not None else theta_cmd.copy()

        def cs_from_theta(th): return np.cos(th), np.sin(th)
        def rotate_cs(c, s, dth):
            cd, sd = np.cos(dth), np.sin(dth)
            c2 = c*cd - s*sd; s2 = s*cd + c*sd
            nrm = np.sqrt(c2*c2 + s2*s2); return c2/nrm, s2/nrm
        def theta_from_cs(c, s): return np.arctan2(s, c)

        def U_grav(th):
            com = pin.centerOfMass(self.model, self.data, th)
            g = self.model.gravity.linear
            return - float(self.total_mass) * float(np.dot(g, com))
        def V_total(th, K_eff):
            d = th - theta_cmd
            return U_grav(th) + 0.5 * float(d.T @ (K_eff @ d))

        it_per_stage = max(1, int(np.ceil(maxiter / max(1, len(lambdas)))))
        self.eq_path_last = []
        c, s = cs_from_theta(theta0)

        for lam in lambdas:
            K_eff = K + (lam * k_stiffness) * np.eye(n)
            mu = float(newton_mu0)
            for _ in range(it_per_stage):
                th = theta_from_cs(c, s)
                tau_g = pin.computeGeneralizedGravity(self.model, self.data, th)
                F = tau_g + K_eff @ (th - theta_cmd)
                dG = pin.computeGeneralizedGravityDerivatives(self.model, self.data, th)
                Jq = dG + K_eff
                if ensure_spd: Jq = 0.5*(Jq + Jq.T) + spd_boost * np.eye(n)

                A = Jq + mu * np.eye(n)
                dth_newton = - (np.linalg.pinv(A, rcond=1e-12) @ F)
                dth_newton = np.clip(dth_newton, -step_clip, step_clip)

                V_cur = V_total(th, K_eff)
                accept = False; dth_try = dth_newton.copy()
                for _bt in range(backtrack_max):
                    c_try, s_try = rotate_cs(c, s, dth_try)
                    th_try = theta_from_cs(c_try, s_try)
                    V_try = V_total(th_try, K_eff)
                    if V_try <= V_cur: accept = True; break
                    dth_try *= backtrack_shrink
                if accept: c, s = c_try, s_try; mu = max(newton_mu0, mu*0.33)
                else:      mu = min(newton_mu_max, mu*3.0)

            self.eq_path_last.append(theta_from_cs(c, s).copy())

        return theta_from_cs(c, s)

    # ---- Kinematics helpers (generic frame) ----
    def get_frame_id(self, frame_name: str) -> int:
        return self.model.getFrameId(frame_name)

    def frame_rotation_in_base(self, theta: np.ndarray, fid: int) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, theta); pin.updateFramePlacements(self.model, self.data)
        R_wb = self.data.oMf[self.base_fid].rotation
        R_wf = self.data.oMf[fid].rotation
        return R_wb.T @ R_wf

    def frame_quaternion_wxyz_base(self, theta: np.ndarray, fid: int) -> np.ndarray:
        R_bf = self.frame_rotation_in_base(theta, fid)
        q_xyzw = Rsc.from_matrix(R_bf).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        return q_wxyz / (np.linalg.norm(q_wxyz) + 1e-18)

    def frame_angular_jacobian_world(self, theta: np.ndarray, fid: int) -> np.ndarray:
        pin.computeJointJacobians(self.model, self.data, theta)
        pin.updateFramePlacements(self.model, self.data)
        J6 = pin.computeFrameJacobian(self.model, self.data, theta, fid, pin.ReferenceFrame.WORLD)
        return J6[3:6, :]

    def gravity_dir_in_frame(self, theta: np.ndarray, g_base: np.ndarray, fid: int) -> np.ndarray:
        R_bf = self.frame_rotation_in_base(theta, fid)
        gb = g_base / (np.linalg.norm(g_base) + 1e-12)
        gf = R_bf @ gb
        return gf / (np.linalg.norm(gf) + 1e-12)

    def fk_pose(self, theta: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, theta); pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tip_fid]

    def joint_transforms(self, theta: np.ndarray) -> List[pin.SE3]:
        pin.forwardKinematics(self.model, self.data, theta); pin.updateFramePlacements(self.model, self.data)
        Ts = [self.data.oMi[jid] for jid in range(1, self.model.njoints)]
        Ts.append(self.data.oMf[self.tip_fid])
        return Ts

# ------------------------ RIK / GaIK ------------------------
class RikGaikPlanner:
    def __init__(self, chain: IkChain):
        self.chain = chain

    def rik_solve(self, T_target: pin.SE3, theta_init: np.ndarray, max_iter: int = 1200) -> np.ndarray:
        T_h = se3_to_homog(T_target)
        q0_full = np.zeros(len(self.chain.links))
        q0_full[1:1+theta_init.size] = theta_init
        sol_full = self.chain.inverse_kinematics_frame(
            T_h, initial_position=q0_full, max_iter=max_iter, orientation_mode="all"
        )
        return np.array(sol_full[1:1+theta_init.size])

    @staticmethod
    def make_theta_ref_path(theta_init: np.ndarray, theta_ref_goal: np.ndarray, n_steps: int) -> np.ndarray:
        a = np.linspace(0.0, 1.0, n_steps)
        return (1-a)[:,None]*theta_init[None,:] + a[:,None]*theta_ref_goal[None,:]

    @staticmethod
    def theta_cmd_from_theta_ref(robot: RobotArm, theta_ref: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        tau_g = pin.computeGeneralizedGravity(robot.model, robot.data, theta_ref)
        return theta_ref + tau_g / np.maximum(kp_vec, 1e-12)

# ------------------------ Observations (multi-frame) ------------------------
class ObservationBuilder:
    def __init__(self, robot_sim: RobotArm, g_base: np.ndarray, parameter_A: float = 100.0):
        self.robot_sim = robot_sim
        self.g_base = np.asarray(g_base, dtype=float)
        self.parameter_A = float(parameter_A)

    def build_A_multi(self, theta_cmd: np.ndarray, kp_true: np.ndarray,
                      frame_names: List[str], newton_iter_true: int = 60,
                      theta_ws_true: Optional[np.ndarray] = None
                      ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Returns:
          A_map: {fid -> A_f}, built from true gravity direction in each frame at equilibrium(theta_cmd, kp_true)
          theta_equil_true: equilibrium used (for warm-start chaining)
        """
        theta_equil_true = self.robot_sim.equilibrium_s1(theta_cmd, kp_true, maxiter=newton_iter_true,
                                                         theta_init=theta_ws_true)
        A_map: Dict[int, np.ndarray] = {}
        for name in frame_names:
            fid = self.robot_sim.get_frame_id(name)
            g_f = self.robot_sim.gravity_dir_in_frame(theta_equil_true, self.g_base, fid)
            A_f = simple_bingham_unit(self.g_base, g_f, parameter=self.parameter_A)
            A_map[fid] = A_f
        return A_map, theta_equil_true

# ------------------------ Multi-frame Weird EKF ------------------------
class MultiFrameWeirdEKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, eps_def: float = 1e-6):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.eps_def = float(eps_def)
        self.last_theta_eq: Optional[np.ndarray] = None

    def predict(self):
        self.P = self.P + self.Q

    def _grad_hess_multi(self, x0: np.ndarray, theta_cmd: np.ndarray,
                         A_map: Dict[int, np.ndarray], robot_est: RobotArm,
                         theta_init: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build gradient g and Hessian H for L(x) = sum_f z_f^T A_f z_f.
        Returns (g, H, theta_eq) at x0.
        """
        n = x0.size
        k = np.exp(x0)
        K = np.diag(k)

        # Equilibrium once (shared for all frames)
        theta_eq = robot_est.equilibrium_s1(theta_cmd, k, maxiter=80, theta_init=theta_init)

        # Common pieces
        dG = pin.computeGeneralizedGravityDerivatives(robot_est.model, robot_est.data, theta_eq)  # n x n
        J_q = dG + K
        J_x = np.diag(k * (theta_eq - theta_cmd))  # n x n

        # Accumulators
        u_total = np.zeros(n)        # sum of J_w^T Q^T A z
        H0_total = np.zeros((n, n))  # sum of 1/2 M^T A M
        MtM_total = np.zeros((n, n))

        for fid, A_f in A_map.items():
            z_f = robot_est.frame_quaternion_wxyz_base(theta_eq, fid)       # 4,
            Qz_f = Q_from_quat_wxyz(z_f)                                     # 4x3
            J_w_f = robot_est.frame_angular_jacobian_world(theta_eq, fid)    # 3 x n

            v_f = Qz_f.T @ (A_f @ z_f)       # 3,
            u_f = J_w_f.T @ v_f              # n,
            u_total += u_f

            # Hessian contrib
            X = np.linalg.pinv(J_q, rcond=1e-12) @ J_x    # n x n
            M_f = Qz_f @ (J_w_f @ X)                      # 4 x n
            H0_f = 0.5 * (M_f.T @ (A_f @ M_f))            # n x n
            H0_total += H0_f
            MtM_total += (M_f.T @ M_f)

        # Gradient
        y = np.linalg.pinv(J_q.T, rcond=1e-12) @ u_total  # solve J_q^T y = u_total
        g = - (J_x.T @ y)

        # Hessian stabilization using Bingham invariance: A_f -> A_f + c I_4
        H0s = 0.5 * (H0_total + H0_total.T)
        wH = np.linalg.eigvalsh(H0s)
        lam_max_H = float(np.max(wH)) if wH.size > 0 else 0.0
        if lam_max_H <= -self.eps_def:
            H = H0s
        else:
            Bs = 0.5 * (MtM_total + MtM_total.T)
            wB = np.linalg.eigvalsh(Bs)
            lam_max_B = float(np.max(wB)) if wB.size > 0 else 0.0
            if lam_max_B <= 1e-12:
                H = H0s - (lam_max_H + self.eps_def) * np.eye(n)
            else:
                c = - 2.0 * (lam_max_H + self.eps_def) / lam_max_B
                H = H0s + 0.5 * c * MtM_total
        return g, H, theta_eq

    def update_with_multi(self, theta_cmd: np.ndarray, A_map: Dict[int, np.ndarray],
                          robot_est: RobotArm, theta_init_eq_pred: Optional[np.ndarray]) -> np.ndarray:
        """
        Perform one Bayes update with multi-frame likelihood.
        Returns the theta_eq used (for warm-start in the next call).
        """
        self.predict()
        g, H, theta_eq = self._grad_hess_multi(self.x, theta_cmd, A_map, robot_est, theta_init_eq_pred)
        Sinv = -H
        # ensure PD
        w = np.linalg.eigvalsh(0.5*(Sinv+Sinv.T)); lam_min = float(np.min(w))
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
        self.last_theta_eq = theta_eq.copy()
        return theta_eq

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
        # for Ti in robot_sim.joint_transforms(theta_cmd_final): self.draw_frame(ax, Ti, axis_len=0.06)
        self.draw_frame(ax, T_target_se3, axis_len=0.08)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title(title); ax.legend()
        all_pts = np.vstack([P_rigid, P_true, P_est, T_target_se3.translation.reshape(1,3)])
        c = all_pts.mean(axis=0); r = max(np.max(np.linalg.norm(all_pts-c, axis=1)), 0.25)
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.view_init(elev=25, azim=45); plt.gca().set_box_aspect([1,1,1]); plt.tight_layout(); plt.show()

# ------------------------ Main (example) ------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    g_base = np.array([0.0, 0.0, -9.81])
    kp_true = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float)
    parameter_A = 100.0
    n_ref_steps = 50  # linear interpolation steps along theta_ref

    # Choose which link frames to use for gravity-direction likelihood
    obs_frames = ["link6"]#, "link3", "link2"]  # <- add/remove names as needed

    # Two arms + IKPy chain
    urdf_path = ensure_urdf()
    robot_sim = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
    robot_est = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
    chain = IkChain.from_urdf_file(urdf_path, base_elements=["base_link"], symbolic=False)

    # Fixed target (made with robot_sim before estimation)
    theta_rand = np.array([rng.uniform(lo, hi)
                           for lo, hi in zip(robot_sim.model.lowerPositionLimit,
                                             robot_sim.model.upperPositionLimit)])
    theta_equil_true_for_target = robot_sim.equilibrium_s1(theta_rand, kp_true, maxiter=80, theta_init=theta_rand)
    T_target_se3 = robot_sim.fk_pose(theta_equil_true_for_target)

    # RIK one-shot to goal; reference path is purely linear in joint space
    planner = RikGaikPlanner(chain)
    theta_init = np.zeros(robot_est.nv)
    theta_ref_goal = planner.rik_solve(T_target_se3, theta_init, max_iter=1200)
    theta_ref_path = planner.make_theta_ref_path(theta_init, theta_ref_goal, n_steps=n_ref_steps)

    # Weird EKF (multi-frame) init
    x0 = np.log(np.ones(robot_est.nv) * 50.0)
    P0 = np.eye(robot_est.nv) * 1.0
    Q  = np.eye(robot_est.nv) * 1e-3
    wekf = MultiFrameWeirdEKF(x0, P0, Q, eps_def=1e-6)

    obs_builder = ObservationBuilder(robot_sim, g_base, parameter_A=parameter_A)

    # Loop over theta_ref path
    theta_ws_true = None
    theta_eq_pred_prev = None
    theta_cmd_final = None

    for k in range(theta_ref_path.shape[0]):
        theta_ref_k = theta_ref_path[k]
        kp_hat = np.exp(wekf.x)

        # GaIK (inverse statics)
        theta_cmd_k = planner.theta_cmd_from_theta_ref(robot_est, theta_ref_k, kp_hat)
        theta_cmd_final = theta_cmd_k

        # Observations for selected frames (robot_sim, truth Kp)
        A_map, theta_equil_true_k = obs_builder.build_A_multi(theta_cmd_k, kp_true, obs_frames,
                                                              newton_iter_true=60, theta_ws_true=theta_ws_true)
        theta_ws_true = theta_equil_true_k

        # Weird EKF update using all frames (warm-start with previous predicted equilibrium or theta_ref_k)
        theta_eq_used = wekf.update_with_multi(theta_cmd_k, A_map, robot_est,
                                               theta_init_eq_pred=(theta_eq_pred_prev if theta_eq_pred_prev is not None
                                                                   else theta_ref_k))
        theta_eq_pred_prev = theta_eq_used

        if (k+1) % max(1, n_ref_steps//4) == 0:
            print(f"[{k+1}/{n_ref_steps}] Kp_hat =", np.exp(wekf.x))

    # Visualization & report
    viz = Visualizer()
    viz.show_triplet(robot_sim, robot_est, theta_cmd_final, kp_true, np.maximum(np.exp(wekf.x),1e-8),
                     T_target_se3, title="Triplet: rigid vs true-gravity vs est-gravity (multi-frame)")

    theta_equil_true_final = robot_sim.equilibrium_s1(theta_cmd_final, kp_true, maxiter=80, theta_init=theta_cmd_final)
    E_final = robot_sim.fk_pose(theta_equil_true_final).inverse() * T_target_se3
    pos_err = np.linalg.norm(E_final.translation)
    rot_err = np.linalg.norm(pin.log3(E_final.rotation))
    print("Kp_true =", kp_true)
    print("Kp_hat  =", np.exp(wekf.x))
    print("pos_err =", pos_err, "rot_err =", rot_err)
