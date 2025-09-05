# Multi-frame gravity-direction likelihood for Weird EKF (sum over selected link frames)
# - RIK: one-shot IKPy to target, then linear interpolation from theta_init to theta_ref_goal
# - GaIK: inverse statics one-shot  theta_cmd = theta_ref + K^{-1} tau_g(theta_ref)
# - Equilibrium solver: energy-minimizing S^1 Newton with mandatory homotopy (lambda linspace 1->0, k_stiffness=100)
# - Observations: for frames {f}, A_f built from robot_sim (truth) gravity direction at equilibrium(theta_cmd, Kp_true)
# - Likelihood: L(x) = sum_f z_f(theta_eq(x; theta_cmd))^T A_f z_f(...)
# - Analytic gradient/Hessian summed over frames; Bingham shift to make -H positive-definite
# - All linear algebra uses pinv (no solve/inv). IK must be IKPy. Variable names are ASCII only.

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc
import matplotlib.pyplot as plt
from ikpy.chain import Chain as IkChain

from scipy.optimize import minimize


# ------------------------ URDF utils ------------------------
def require_urdf(urdf_path: str = "simple6r.urdf") -> str:
    """
    Ensure the URDF file exists. (No generation here.)
    """
    abs_path = os.path.abspath(urdf_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"URDF not found: {abs_path}")
    return abs_path


# ------------------------ Math / Geometry utils ------------------------
def se3_to_homog(M: pin.SE3) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = M.rotation
    T[:3, 3] = M.translation
    return T


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n


# ------------------------ Bingham helpers (NO normalization const) ------------------------
class BinghamUtils:
    @staticmethod
    def qmat_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array(
            [
                [-x, -y, -z],
                [w, -z, y],
                [z, w, -x],
                [-y, x, w],
            ],
            dtype=float,
        )

    @staticmethod
    def _lmat(q: np.ndarray) -> np.ndarray:
        a, b, c, d = q
        return np.array(
            [
                [a, -b, -c, -d],
                [b, a, -d, c],
                [c, d, a, -b],
                [d, -c, b, a],
            ],
            dtype=float,
        )

    @staticmethod
    def _rmat(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array(
            [
                [w, -x, -y, -z],
                [x, w, z, -y],
                [y, -z, w, x],
                [z, y, -x, w],
            ],
            dtype=float,
        )

    @staticmethod
    def simple_bingham_unit(
        before_vec3: np.ndarray, after_vec3: np.ndarray, parameter: float = 100.0
    ) -> np.ndarray:
        b = _normalize(np.asarray(before_vec3, dtype=float))
        a = _normalize(np.asarray(after_vec3, dtype=float))
        vq = np.array([0.0, b[0], b[1], b[2]], dtype=float)
        xq = np.array([0.0, a[0], a[1], a[2]], dtype=float)
        P = BinghamUtils._lmat(xq) - BinghamUtils._rmat(vq)
        A0 = -0.25 * (P.T @ P)  # semi-negative definite
        return float(parameter) * A0


# ------------------------ Robot wrapper ------------------------
class RobotArm:
    def __init__(self, urdf_path: str, tip_link: str = "link6", base_link: str = "base_link") -> None:
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.tip_fid = self.model.getFrameId(tip_link)
        self.base_fid = self.model.getFrameId(base_link)
        if hasattr(self.model, "gravity"):
            self.model.gravity.linear = np.array([0.0, 0.0, -9.81], dtype=float)
        self.total_mass = float(sum(inert.mass for inert in self.model.inertias))

    # ---- Kinematics helpers (generic frame) ----
    def get_frame_id(self, frame_name: str) -> int:
        return self.model.getFrameId(frame_name)

    def _fk_update(self, theta: np.ndarray) -> None:
        pin.forwardKinematics(self.model, self.data, theta)
        pin.updateFramePlacements(self.model, self.data)

    def frame_rotation_in_base(self, theta: np.ndarray, fid: int) -> np.ndarray:
        self._fk_update(theta)
        R_wb = self.data.oMf[self.base_fid].rotation
        R_wf = self.data.oMf[fid].rotation
        return R_wb.T @ R_wf

    def frame_quaternion_wxyz_base(self, theta: np.ndarray, fid: int) -> np.ndarray:
        R_bf = self.frame_rotation_in_base(theta, fid)
        q_xyzw = Rsc.from_matrix(R_bf).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
        return _normalize(q_wxyz, eps=1e-18)

    def frame_angular_jacobian_world(self, theta: np.ndarray, fid: int) -> np.ndarray:
        pin.computeJointJacobians(self.model, self.data, theta)
        pin.updateFramePlacements(self.model, self.data)
        J6 = pin.computeFrameJacobian(self.model, self.data, theta, fid, pin.ReferenceFrame.WORLD)
        return J6[3:6, :]

    def gravity_dir_in_frame(self, theta: np.ndarray, g_base: np.ndarray, fid: int) -> np.ndarray:
        R_bf = self.frame_rotation_in_base(theta, fid)
        gb = _normalize(g_base)
        gf = R_bf @ gb
        return _normalize(gf)

    def fk_pose(self, theta: np.ndarray) -> pin.SE3:
        self._fk_update(theta)
        M = self.data.oMf[self.tip_fid]
        # ディープコピー（参照を返さない）
        return pin.SE3(M)  # これで内部バッファから独立

    def joint_transforms(self, theta: np.ndarray) -> List[pin.SE3]:
        self._fk_update(theta)
        Ts = [pin.SE3(self.data.oMi[jid]) for jid in range(1, self.model.njoints)]
        Ts.append(pin.SE3(self.data.oMf[self.tip_fid]))
        return Ts


    # ---- Gravity & derivatives ----
    def tau_gravity(self, theta: np.ndarray) -> np.ndarray:
        return pin.computeGeneralizedGravity(self.model, self.data, theta)

    def d_tau_gravity(self, theta: np.ndarray) -> np.ndarray:
        return pin.computeGeneralizedGravityDerivatives(self.model, self.data, theta)

    def potential_gravity(self, theta: np.ndarray) -> float:
        com = pin.centerOfMass(self.model, self.data, theta)
        g = self.model.gravity.linear
        return -self.total_mass * float(np.dot(g, com))


# ------------------------ Equilibrium solver (S^1 Newton with homotopy) ------------------------
@dataclass
class EquilibriumConfig:
    maxiter: int = 200            # per lambda stage
    k_stiffness: float = 100.0    # homotopy ridge
    n_lambda: int = 10            # number of homotopy stages
    ftol: float = 1e-9            # optimizer tolerance
    verbose: bool = False         # SLSQP display

class EquilibriumSolver:
    def __init__(self, cfg: Optional[EquilibriumConfig] = None) -> None:
        self.cfg = cfg or EquilibriumConfig()
        self.eq_path_last: List[np.ndarray] = []

    # ---------- helpers (cos-sin parameterization) ----------
    @staticmethod
    def _pack_cs(c: np.ndarray, s: np.ndarray) -> np.ndarray:
        # interleaved: [c0, s0, c1, s1, ...]
        out = np.empty(c.size * 2, dtype=float)
        out[0::2] = c
        out[1::2] = s
        return out

    @staticmethod
    def _unpack_cs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = x[0::2]
        s = x[1::2]
        return c, s

    @staticmethod
    def _theta_from_cs(c: np.ndarray, s: np.ndarray) -> np.ndarray:
        return np.arctan2(s, c)

    # ---------- objective & gradient in x = (c,s) ----------
    @staticmethod
    def _V_total(robot, theta: np.ndarray, theta_cmd: np.ndarray, k_eff_diag: np.ndarray) -> float:
        # U(theta) + 1/2 * (theta - theta_cmd)^T K_eff (theta - theta_cmd)
        U = robot.potential_gravity(theta)
        d = theta - theta_cmd
        return float(U + 0.5 * np.dot(d * k_eff_diag, d))

    @staticmethod
    def _grad_theta(robot, theta: np.ndarray, theta_cmd: np.ndarray, k_eff_diag: np.ndarray) -> np.ndarray:
        # ∂V/∂theta = tau_g(theta) + K_eff (theta - theta_cmd)
        tau_g = robot.tau_gravity(theta)
        return tau_g + k_eff_diag * (theta - theta_cmd)

    @staticmethod
    def _grad_x_from_grad_theta(g_theta: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
        # theta_i = atan2(s_i, c_i)
        # dtheta/dc = -s / (c^2 + s^2), dtheta/ds =  c / (c^2 + s^2)
        denom = np.maximum(c * c + s * s, 1e-12)
        dtheta_dc = -s / denom
        dtheta_ds =  c / denom
        gx = np.empty(g_theta.size * 2, dtype=float)
        gx[0::2] = g_theta * dtheta_dc
        gx[1::2] = g_theta * dtheta_ds
        return gx

    # ---------- constraints c_i^2 + s_i^2 = 1 ----------
    @staticmethod
    def _cons_fun(x: np.ndarray) -> np.ndarray:
        c, s = EquilibriumSolver._unpack_cs(x)
        return c * c + s * s - 1.0

    @staticmethod
    def _cons_jac(x: np.ndarray) -> np.ndarray:
        c, s = EquilibriumSolver._unpack_cs(x)
        n = c.size
        J = np.zeros((n, 2 * n), dtype=float)
        idx = np.arange(n)
        J[idx, 2 * idx] = 2.0 * c
        J[idx, 2 * idx + 1] = 2.0 * s
        return J

    def _stage_minimize(
        self,
        robot,
        theta_cmd: np.ndarray,
        k_eff_diag: np.ndarray,
        x0: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One SLSQP solve for fixed K_eff (one homotopy stage).
        Returns (x_opt, theta_opt).
        """
        def f_obj(x: np.ndarray) -> float:
            c, s = self._unpack_cs(x)
            theta = self._theta_from_cs(c, s)
            return self._V_total(robot, theta, theta_cmd, k_eff_diag)

        def f_jac(x: np.ndarray) -> np.ndarray:
            c, s = self._unpack_cs(x)
            theta = self._theta_from_cs(c, s)
            g_theta = self._grad_theta(robot, theta, theta_cmd, k_eff_diag)
            return self._grad_x_from_grad_theta(g_theta, c, s)

        cons = {
            "type": "eq",
            "fun": self._cons_fun,
            "jac": self._cons_jac,
        }

        res = minimize(
            fun=f_obj,
            x0=x0,
            jac=f_jac,
            constraints=[cons],
            method="SLSQP",
            options={
                "maxiter": int(self.cfg.maxiter),
                "ftol": float(self.cfg.ftol),
                "disp": bool(self.cfg.verbose),
            },
        )
        x_opt = res.x
        c_opt, s_opt = self._unpack_cs(x_opt)
        theta_opt = self._theta_from_cs(c_opt, s_opt)
        return x_opt, theta_opt

    def solve(
        self,
        robot,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
        theta_init: Optional[np.ndarray] = None,
        lambdas: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Homotopy over lambda in [1 -> 0]. Each stage solves:
            min_{c,s} U(theta(c,s)) + 1/2 (theta - theta_cmd)^T K_eff(theta - theta_cmd)
            s.t. c_i^2 + s_i^2 = 1
        """
        if lambdas is None:
            lambdas = np.linspace(1.0, 0.0, self.cfg.n_lambda)

        n = kp_vec.size
        theta0 = theta_init.copy() if theta_init is not None else theta_cmd.copy()
        c0 = np.cos(theta0)
        s0 = np.sin(theta0)
        x0 = self._pack_cs(c0, s0)

        self.eq_path_last = []

        for lam in lambdas:
            k_eff_diag = kp_vec + float(lam) * float(self.cfg.k_stiffness)
            x0, theta_opt = self._stage_minimize(robot, theta_cmd, k_eff_diag, x0)
            self.eq_path_last.append(theta_opt.copy())

        return theta_opt

# ------------------------ RIK / GaIK ------------------------
class RikGaikPlanner:
    def __init__(self, chain: IkChain) -> None:
        self.chain = chain

    def rik_solve(self, T_target: pin.SE3, theta_init: np.ndarray, max_iter: int = 1200) -> np.ndarray:
        T_h = se3_to_homog(T_target)
        q0_full = np.zeros(len(self.chain.links), dtype=float)
        q0_full[1 : 1 + theta_init.size] = theta_init
        sol_full = self.chain.inverse_kinematics_frame(
            T_h, initial_position=q0_full, max_iter=max_iter, orientation_mode="all"
        )
        return np.array(sol_full[1 : 1 + theta_init.size], dtype=float)

    @staticmethod
    def make_theta_ref_path(theta_init: np.ndarray, theta_ref_goal: np.ndarray, n_steps: int) -> np.ndarray:
        a = np.linspace(0.0, 1.0, n_steps)
        return (1 - a)[:, None] * theta_init[None, :] + a[:, None] * theta_ref_goal[None, :]

    @staticmethod
    def theta_cmd_from_theta_ref(robot: RobotArm, theta_ref: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        tau_g = robot.tau_gravity(theta_ref)
        kp_safe = np.maximum(kp_vec, 1e-12)
        return theta_ref + (tau_g / kp_safe)


# ------------------------ Observations (multi-frame) ------------------------
class ObservationBuilder:
    def __init__(self, robot_sim: RobotArm, g_base: np.ndarray, parameter_A: float = 100.0) -> None:
        self.robot_sim = robot_sim
        self.g_base = np.asarray(g_base, dtype=float)
        self.parameter_A = float(parameter_A)

    def build_A_multi(
        self,
        solver: EquilibriumSolver,
        theta_cmd: np.ndarray,
        kp_true: np.ndarray,
        frame_names: List[str],
        newton_iter_true: int = 60,
        theta_ws_true: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Returns:
          A_map: {fid -> A_f}, built from true gravity direction in each frame at equilibrium(theta_cmd, kp_true)
          theta_equil_true: equilibrium used (for warm-start chaining)
        """
        cfg = EquilibriumConfig(maxiter=newton_iter_true)
        tmp_solver = EquilibriumSolver(cfg=cfg)
        theta_equil_true = tmp_solver.solve(
            robot=self.robot_sim,
            theta_cmd=theta_cmd,
            kp_vec=kp_true,
            theta_init=theta_ws_true,
        )

        A_map: Dict[int, np.ndarray] = {}
        for name in frame_names:
            fid = self.robot_sim.get_frame_id(name)
            g_f = self.robot_sim.gravity_dir_in_frame(theta_equil_true, self.g_base, fid)
            A_f = BinghamUtils.simple_bingham_unit(self.g_base, g_f, parameter=self.parameter_A)
            A_map[fid] = A_f

        return A_map, theta_equil_true


# ------------------------ Multi-frame Weird EKF ------------------------
class MultiFrameWeirdEKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, eps_def: float = 1e-6) -> None:
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.eps_def = float(eps_def)
        self.last_theta_eq: Optional[np.ndarray] = None

    def predict(self) -> None:
        self.P = self.P + self.Q

    @staticmethod
    def _common_terms(
        robot: RobotArm, theta_eq: np.ndarray, theta_cmd: np.ndarray, k_diag: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        dG = robot.d_tau_gravity(theta_eq)
        K = np.diag(k_diag)
        J_q = dG + K
        J_x = np.diag(k_diag * (theta_eq - theta_cmd))
        return J_q, J_x

    def _accumulate_frame_terms(
        self,
        robot: RobotArm,
        theta_eq: np.ndarray,
        fid: int,
        A_f: np.ndarray,
        J_q: np.ndarray,
        J_x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_f = robot.frame_quaternion_wxyz_base(theta_eq, fid)  # (4,)
        Qz_f = BinghamUtils.qmat_from_quat_wxyz(z_f)           # (4,3)
        J_w_f = robot.frame_angular_jacobian_world(theta_eq, fid)  # (3, n)

        v_f = Qz_f.T @ (A_f @ z_f)   # (3,)
        u_f = J_w_f.T @ v_f          # (n,)

        X = np.linalg.pinv(J_q, rcond=1e-12) @ J_x             # (n, n)
        M_f = Qz_f @ (J_w_f @ X)                               # (4, n)
        H0_f = 0.5 * (M_f.T @ (A_f @ M_f))
        MtM_f = M_f.T @ M_f
        return u_f, H0_f, MtM_f

    def _stabilize_hessian(self, H0_total: np.ndarray, MtM_total: np.ndarray) -> np.ndarray:
        H0s = 0.5 * (H0_total + H0_total.T)
        wH = np.linalg.eigvalsh(H0s)
        lam_max_H = float(np.max(wH)) if wH.size > 0 else 0.0
        if lam_max_H <= -self.eps_def:
            return H0s

        Bs = 0.5 * (MtM_total + MtM_total.T)
        wB = np.linalg.eigvalsh(Bs)
        lam_max_B = float(np.max(wB)) if wB.size > 0 else 0.0

        if lam_max_B <= 1e-12:
            return H0s - (lam_max_H + self.eps_def) * np.eye(H0s.shape[0])

        c = -2.0 * (lam_max_H + self.eps_def) / lam_max_B
        return H0s + 0.5 * c * MtM_total

    def _grad_hess_multi(
        self,
        solver: EquilibriumSolver,
        x0: np.ndarray,
        theta_cmd: np.ndarray,
        A_map: Dict[int, np.ndarray],
        robot_est: RobotArm,
        theta_init: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = x0.size
        k_diag = np.exp(x0)

        theta_eq = solver.solve(
            robot=robot_est, theta_cmd=theta_cmd, kp_vec=k_diag, theta_init=theta_init
        )

        J_q, J_x = self._common_terms(robot_est, theta_eq, theta_cmd, k_diag)

        u_total = np.zeros(n, dtype=float)
        H0_total = np.zeros((n, n), dtype=float)
        MtM_total = np.zeros((n, n), dtype=float)

        for fid, A_f in A_map.items():
            u_f, H0_f, MtM_f = self._accumulate_frame_terms(robot_est, theta_eq, fid, A_f, J_q, J_x)
            u_total += u_f
            H0_total += H0_f
            MtM_total += MtM_f

        y = np.linalg.pinv(J_q.T, rcond=1e-12) @ u_total  # solve J_q^T y = u_total
        g = -(J_x.T @ y)

        H = self._stabilize_hessian(H0_total, MtM_total)
        return g, H, theta_eq

    def update_with_multi(
        self,
        solver: EquilibriumSolver,
        theta_cmd: np.ndarray,
        A_map: Dict[int, np.ndarray],
        robot_est: RobotArm,
        theta_init_eq_pred: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Perform one Bayes update with multi-frame likelihood.
        Returns the theta_eq used (for warm-start in the next call).
        """
        self.predict()
        g, H, theta_eq = self._grad_hess_multi(
            solver=solver,
            x0=self.x,
            theta_cmd=theta_cmd,
            A_map=A_map,
            robot_est=robot_est,
            theta_init=theta_init_eq_pred,
        )
        Sinv = -H
        w = np.linalg.eigvalsh(0.5 * (Sinv + Sinv.T))
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

        self.P = 0.5 * (P_post + P_post.T)
        self.x = x_post
        self.last_theta_eq = theta_eq.copy()
        return theta_eq


# ------------------------ Visualization ------------------------
class Visualizer:
    @staticmethod
    def draw_frame(ax, T: pin.SE3, axis_len: float = 0.06) -> None:
        o = T.translation
        Rm = T.rotation
        cols = ["r", "g", "b"]
        for k in range(3):
            a = Rm[:, k] * axis_len
            ax.plot([o[0], o[0] + a[0]], [o[1], o[1] + a[1]], [o[2], o[2] + a[2]], cols[k], linewidth=1.8)
        ax.scatter([o[0]], [o[1]], [o[2]], s=35, c="k")

    @staticmethod
    def joint_positions(robot: RobotArm, theta: np.ndarray) -> np.ndarray:
        robot._fk_update(theta)
        pts = [robot.data.oMi[jid].translation for jid in range(1, robot.model.njoints)]
        pts.append(robot.data.oMf[robot.tip_fid].translation)
        return np.vstack(pts)

    def show_triplet(
        self,
        solver: EquilibriumSolver,
        robot_sim: RobotArm,
        robot_est: RobotArm,
        theta_cmd_final: np.ndarray,
        kp_true: np.ndarray,
        kp_est: np.ndarray,
        T_target_se3: pin.SE3,
        title: str = "Final comparison",
    ) -> None:
        P_rigid = self.joint_positions(robot_sim, theta_cmd_final)

        theta_equil_true = EquilibriumSolver(EquilibriumConfig(maxiter=80)).solve(
            robot_sim, theta_cmd_final, kp_true, theta_init=theta_cmd_final
        )
        P_true = self.joint_positions(robot_sim, theta_equil_true)

        theta_equil_est = EquilibriumSolver(EquilibriumConfig(maxiter=80)).solve(
            robot_est, theta_cmd_final, np.maximum(kp_est, 1e-8), theta_init=theta_cmd_final
        )
        P_est = self.joint_positions(robot_est, theta_equil_est)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(P_rigid[:, 0], P_rigid[:, 1], P_rigid[:, 2], "o--", label="sim rigid (theta_cmd)")
        ax.plot(P_true[:, 0], P_true[:, 1], P_true[:, 2], "o-", label="sim gravity (Kp_true)")
        ax.plot(P_est[:, 0], P_est[:, 1], P_est[:, 2], "o-.", label="est gravity (Kp_est)")
        self.draw_frame(ax, T_target_se3, axis_len=0.08)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.legend()

        all_pts = np.vstack([P_rigid, P_true, P_est, T_target_se3.translation.reshape(1, 3)])
        c = all_pts.mean(axis=0)
        r = max(np.max(np.linalg.norm(all_pts - c, axis=1)), 0.25)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.view_init(elev=25, azim=45)
        plt.gca().set_box_aspect([1, 1, 1])
        plt.tight_layout()
        plt.show()


# ------------------------ Main (example) ------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(9)
    g_base = np.array([0.0, 0.0, -9.81], dtype=float)
    kp_true = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float) * 2
    parameter_A = 100.0
    n_ref_steps = 50  # linear interpolation steps along theta_ref

    # Choose which link frames to use for gravity-direction likelihood
    obs_frames = ["link6", "link3", "link4"]  # <- add/remove names as needed

    # Two arms + IKPy chain
    urdf_path = require_urdf("simple6r.urdf")
    robot_sim = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
    robot_est = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
    chain = IkChain.from_urdf_file(urdf_path, base_elements=["base_link"], symbolic=False)

    # Fixed target (made with robot_sim before estimation)
    theta_rand = np.array(
        [
            rng.uniform(lo, hi)
            for lo, hi in zip(robot_sim.model.lowerPositionLimit, robot_sim.model.upperPositionLimit)
        ],
        dtype=float,
    )

    solver_default = EquilibriumSolver(EquilibriumConfig(maxiter=80))
    theta_equil_true_for_target = solver_default.solve(
        robot=robot_sim, theta_cmd=theta_rand, kp_vec=kp_true, theta_init=theta_rand
    )
    T_target_se3 = robot_sim.fk_pose(theta_equil_true_for_target)

    # RIK one-shot to goal; reference path is purely linear in joint space
    planner = RikGaikPlanner(chain)
    theta_init = np.zeros(robot_est.nv, dtype=float)
    theta_ref_goal = planner.rik_solve(T_target_se3, theta_init, max_iter=1200)
    theta_ref_path = planner.make_theta_ref_path(theta_init, theta_ref_goal, n_steps=n_ref_steps)

    # Weird EKF (multi-frame) init
    x0 = np.log(np.ones(robot_est.nv) * 50.0)
    P0 = np.eye(robot_est.nv) * 1.0
    Q = np.eye(robot_est.nv) * 1e-3
    wekf = MultiFrameWeirdEKF(x0, P0, Q, eps_def=1e-6)

    obs_builder = ObservationBuilder(robot_sim, g_base, parameter_A=parameter_A)

    # Loop over theta_ref path
    theta_ws_true: Optional[np.ndarray] = None
    theta_eq_pred_prev: Optional[np.ndarray] = None
    theta_cmd_final: Optional[np.ndarray] = None

    for k in range(theta_ref_path.shape[0]):
        theta_ref_k = theta_ref_path[k]
        kp_hat = np.exp(wekf.x)

        # GaIK (inverse statics)
        theta_cmd_k = planner.theta_cmd_from_theta_ref(robot_est, theta_ref_k, kp_hat)
        theta_cmd_final = theta_cmd_k

        # Observations for selected frames (robot_sim, truth Kp)
        A_map, theta_equil_true_k = obs_builder.build_A_multi(
            solver=solver_default,
            theta_cmd=theta_cmd_k,
            kp_true=kp_true,
            frame_names=obs_frames,
            newton_iter_true=60,
            theta_ws_true=theta_ws_true,
        )
        theta_ws_true = theta_equil_true_k

        # Weird EKF update using all frames (warm-start with previous predicted equilibrium or theta_ref_k)
        theta_eq_used = wekf.update_with_multi(
            solver=solver_default,
            theta_cmd=theta_cmd_k,
            A_map=A_map,
            robot_est=robot_est,
            theta_init_eq_pred=(theta_eq_pred_prev if theta_eq_pred_prev is not None else theta_ref_k),
        )
        theta_eq_pred_prev = theta_eq_used

        if (k + 1) % max(1, n_ref_steps // 4) == 0:
            print(f"[{k+1}/{n_ref_steps}] Kp_hat =", np.exp(wekf.x))

    # Visualization & report
    viz = Visualizer()
    assert theta_cmd_final is not None
    viz.show_triplet(
        solver=solver_default,
        robot_sim=robot_sim,
        robot_est=robot_est,
        theta_cmd_final=theta_cmd_final,
        kp_true=kp_true,
        kp_est=np.maximum(np.exp(wekf.x), 1e-8),
        T_target_se3=T_target_se3,
        title="Triplet: rigid vs true-gravity vs est-gravity (multi-frame)",
    )

    theta_equil_true_final = solver_default.solve(
        robot=robot_sim, theta_cmd=theta_cmd_final, kp_vec=kp_true, theta_init=theta_cmd_final
    )
    E_final = robot_sim.fk_pose(theta_equil_true_final).inverse() * T_target_se3
    pos_err = np.linalg.norm(E_final.translation)
    rot_err = np.linalg.norm(pin.log3(E_final.rotation))
    print("Kp_true =", kp_true)
    print("Kp_hat  =", np.exp(wekf.x))
    print("pos_err =", pos_err, "rot_err =", rot_err)
