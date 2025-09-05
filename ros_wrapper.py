#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 wrapper node for Multi-frame gravity-direction Weird EKF.
Inputs:
  - /urdf_xml (std_msgs/String)             : URDF XML (受信時に Pinocchio を再構築)
    * もしくは param ~urdf_path or /robot_description を使用
  - /theta_ref (trajectory_msgs/JointTrajectory) : 外部IKの関節角度列（時間順）
  - /imu topics (sensor_msgs/Imu, list)     : IMU 計測（linear_acceleration）frame_id 必須
  - tf (tf2_ros)                            : IMU frame -> link frame の姿勢

Outputs:
  - /theta_cmd (trajectory_msgs/JointTrajectory) : 補正後 関節角度列
  - /kp_estimate (std_msgs/Float64MultiArray)    : 推定 Kp (exp(x))
  - /kp_covariance (std_msgs/Float64MultiArray)  : 共分散行列（row-major）
"""

from typing import Optional, Dict, List, Tuple
import os
import tempfile
import threading
import numpy as np
import rospy
from std_msgs.msg import String, Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Imu
import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as Rsc

# === import your core classes (adjust module name if needed) ===
from weird_ekf_core import (
    RobotArm,
    EquilibriumSolver,
    EquilibriumConfig,
    MultiFrameWeirdEKF,
    BinghamUtils,
)

# --------------------- small helpers ---------------------
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < eps else (v / n)

def quat_to_rotm(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return Rsc.from_quat([qx, qy, qz, qw]).as_matrix()

def layout_for_matrix(rows: int, cols: int) -> MultiArrayLayout:
    return MultiArrayLayout(
        dim=[
            MultiArrayDimension(label="rows", size=rows, stride=rows * cols),
            MultiArrayDimension(label="cols", size=cols, stride=cols),
        ],
        data_offset=0,
    )

# --------------------- IMU buffer (per link) ---------------------
class ImuBuffer:
    def __init__(
        self,
        tf_buffer: tf2_ros.Buffer,
        imu_to_link_map: Dict[str, str],
        invert_acc: bool = True,
    ) -> None:
        self.tf_buffer = tf_buffer
        self.imu_to_link_map = imu_to_link_map
        self.invert_acc = invert_acc
        self.lock = threading.Lock()
        self.link_g_map: Dict[str, np.ndarray] = {}

    def update_from_msg(self, msg: Imu) -> None:
        imu_frame = msg.header.frame_id
        if not imu_frame:
            return
        link_frame = self.imu_to_link_map.get(imu_frame, None)
        if link_frame is None:
            # allow 1:1 if IMU frame == link frame
            link_frame = imu_frame
        try:
            # transform from imu -> link
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame=link_frame,
                source_frame=imu_frame,
                time=rospy.Time(0),
                timeout=rospy.Duration(0.2),
            )
        except Exception:
            return

        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        g_imu = np.array([ax, ay, az], dtype=float)
        if self.invert_acc:
            g_imu = -g_imu  # treat as gravity direction

        # rotate vector into link frame using transform rotation
        q = tf_msg.transform.rotation
        R_li = quat_to_rotm(q.x, q.y, q.z, q.w)
        g_link = normalize(R_li @ g_imu)

        with self.lock:
            self.link_g_map[link_frame] = g_link

    def get_link_g(self, link_frame: str) -> Optional[np.ndarray]:
        with self.lock:
            return self.link_g_map.get(link_frame, None)

# --------------------- A-matrix builder from live IMU ---------------------
class LiveObservationBuilder:
    def __init__(self, g_base: np.ndarray, parameter_A: float) -> None:
        self.g_base = normalize(np.asarray(g_base, dtype=float))
        self.parameter_A = float(parameter_A)

    def build_A_map_from_imu(
        self,
        imu_buf: ImuBuffer,
        obs_links: List[str],
    ) -> Dict[int, np.ndarray]:
        # Placeholder; frame ids (ints) are resolved later by RobotArm
        raise NotImplementedError


# --------------------- ROS Node ---------------------
class WeirdEkfRosNode:
    def __init__(self) -> None:
        # ---- params ----
        self.base_link: str = rospy.get_param("~base_link", "base_link")
        self.tip_link: str = rospy.get_param("~tip_link", "link6")
        self.obs_links: List[str] = rospy.get_param("~obs_links", ["link6", "link3", "link2"])
        self.imu_topics: List[str] = rospy.get_param("~imu_topics", ["/imu"])
        self.imu_to_link_map: Dict[str, str] = rospy.get_param("~imu_to_link_map", {})  # imu_frame -> link_frame
        self.imu_invert_acc: bool = rospy.get_param("~imu_invert_acc", True)
        self.parameter_A: float = float(rospy.get_param("~parameter_A", 100.0))
        self.g_base: np.ndarray = np.asarray(rospy.get_param("~g_base", [0.0, 0.0, -9.81]), dtype=float)

        # solver / filter configs
        maxiter_eq: int = int(rospy.get_param("~equilibrium_maxiter", 80))
        self.solver = EquilibriumSolver(EquilibriumConfig(maxiter=maxiter_eq))
        nv_default: int = int(rospy.get_param("~nv_default", 6))
        kp_init_scalar: float = float(rospy.get_param("~kp_init_scalar", 50.0))
        P0_scalar: float = float(rospy.get_param("~P0_scalar", 1.0))
        Q_scalar: float = float(rospy.get_param("~Q_scalar", 1e-3))
        self.eps_def: float = float(rospy.get_param("~eps_def", 1e-6))

        # RobotArm will be (re)built when URDF arrives or via param
        self.robot: Optional[RobotArm] = None
        self.actuated_joint_names: List[str] = []

        # filter state (initialized after robot is ready)
        self.wekf: Optional[MultiFrameWeirdEKF] = None
        self.theta_eq_pred_prev: Optional[np.ndarray] = None

        # tf & IMU buffer
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.imu_buf = ImuBuffer(self.tf_buffer, self.imu_to_link_map, invert_acc=self.imu_invert_acc)

        # pubs
        self.pub_cmd = rospy.Publisher("theta_cmd", JointTrajectory, queue_size=1)
        self.pub_kp = rospy.Publisher("kp_estimate", Float64MultiArray, queue_size=1)
        self.pub_cov = rospy.Publisher("kp_covariance", Float64MultiArray, queue_size=1)

        # subs
        self.sub_urdf = rospy.Subscriber("urdf_xml", String, self.on_urdf)
        self.sub_traj = rospy.Subscriber("theta_ref", JointTrajectory, self.on_theta_ref)

        # imu subscribers (plural)
        self.imu_subs = [rospy.Subscriber(topic, Imu, self.on_imu) for topic in self.imu_topics]

        # Try initial URDF from params
        self.try_init_robot_from_params(
            urdf_path=rospy.get_param("~urdf_path", ""),
            use_robot_description=rospy.get_param("~use_robot_description", True),
            nv_default=nv_default,
            kp_init_scalar=kp_init_scalar,
            P0_scalar=P0_scalar,
            Q_scalar=Q_scalar,
        )

    # --------------- robot init / rebuild ---------------
    def try_init_robot_from_params(
        self,
        urdf_path: str,
        use_robot_description: bool,
        nv_default: int,
        kp_init_scalar: float,
        P0_scalar: float,
        Q_scalar: float,
    ) -> None:
        xml: Optional[str] = None
        path: Optional[str] = None

        if urdf_path and os.path.exists(urdf_path):
            path = urdf_path
        elif use_robot_description and rospy.has_param("/robot_description"):
            xml = rospy.get_param("/robot_description")
        else:
            rospy.logwarn("URDF not provided yet (topic or params). Waiting for /urdf_xml...")

        if xml is not None:
            path = self.write_urdf_to_tmp(xml)

        if path is None:
            # not ready yet
            self.init_filter_placeholder(nv_default, kp_init_scalar, P0_scalar, Q_scalar)
            return

        self.init_robot_and_filter(path, kp_init_scalar, P0_scalar, Q_scalar)

    def write_urdf_to_tmp(self, xml: str) -> str:
        tmp = tempfile.NamedTemporaryFile(prefix="weird_ekf_", suffix=".urdf", delete=False)
        tmp.write(xml.encode("utf-8"))
        tmp.flush()
        tmp.close()
        rospy.loginfo(f"URDF written to {tmp.name}")
        return tmp.name

    def init_filter_placeholder(self, nv_default: int, kp_init_scalar: float, P0_scalar: float, Q_scalar: float) -> None:
        # create placeholder filter dims (before URDF)
        x0 = np.log(np.ones(nv_default) * kp_init_scalar)
        P0 = np.eye(nv_default) * P0_scalar
        Q = np.eye(nv_default) * Q_scalar
        self.wekf = MultiFrameWeirdEKF(x0, P0, Q, eps_def=self.eps_def)

    def init_robot_and_filter(self, urdf_path: str, kp_init_scalar: float, P0_scalar: float, Q_scalar: float) -> None:
        self.robot = RobotArm(urdf_path, tip_link=self.tip_link, base_link=self.base_link)
        # joint name order from model
        self.actuated_joint_names = [self.robot.model.names[i] for i in range(1, self.robot.model.njoints)]
        nv = self.robot.nv

        # init filter
        x0 = np.log(np.ones(nv) * kp_init_scalar)
        P0 = np.eye(nv) * P0_scalar
        Q = np.eye(nv) * Q_scalar
        self.wekf = MultiFrameWeirdEKF(x0, P0, Q, eps_def=self.eps_def)
        self.theta_eq_pred_prev = None

        rospy.loginfo(f"Robot initialized: nv={nv}, joints={self.actuated_joint_names}")

    # --------------- callbacks ---------------
    def on_urdf(self, msg: String) -> None:
        path = self.write_urdf_to_tmp(msg.data)
        self.init_robot_and_filter(
            urdf_path=path,
            kp_init_scalar=float(np.exp(self.wekf.x[0])) if self.wekf is not None else 50.0,
            P0_scalar=float(np.mean(np.diag(self.wekf.P))) if (self.wekf is not None) else 1.0,
            Q_scalar=float(np.mean(np.diag(self.wekf.Q))) if (self.wekf is not None) else 1e-3,
        )

    def on_imu(self, msg: Imu) -> None:
        self.imu_buf.update_from_msg(msg)

    def on_theta_ref(self, msg: JointTrajectory) -> None:
        if self.robot is None or self.wekf is None:
            rospy.logwarn("Robot or filter not ready; ignoring theta_ref.")
            return

        # Build output trajectory with same header/joint_names/timing
        out = JointTrajectory()
        out.header = msg.header
        out.joint_names = msg.joint_names

        # Process each point sequentially (state carried inside self.wekf)
        for k, pt in enumerate(msg.points):
            theta_ref = self.extract_theta_from_point(msg.joint_names, pt)
            kp_hat = np.exp(self.wekf.x)

            # GaIK (inverse statics) one-shot
            tau_g = self.robot.tau_gravity(theta_ref)
            theta_cmd = theta_ref + tau_g / np.maximum(kp_hat, 1e-12)

            # Build A_map from latest IMU readings on observation links
            A_map, valid_links = self.make_A_map_from_imu()

            # Update EKF if we have any observations
            if A_map:
                theta_init_eq = self.theta_eq_pred_prev if self.theta_eq_pred_prev is not None else theta_ref
                theta_eq_used = self.wekf.update_with_multi(
                    solver=self.solver,
                    theta_cmd=theta_cmd,
                    A_map=A_map,
                    robot_est=self.robot,
                    theta_init_eq_pred=theta_init_eq,
                )
                self.theta_eq_pred_prev = theta_eq_used
            else:
                # no obs -> predict only
                self.wekf.predict()
                rospy.logwarn_throttle(2.0, "No IMU observations available on obs_links; predict-only step.")

            # Append theta_cmd to trajectory (keep original timing)
            out_pt = JointTrajectoryPoint()
            out_pt.positions = [theta_cmd[self.index_of_joint(n)] for n in msg.joint_names]
            out_pt.time_from_start = pt.time_from_start
            out.points.append(out_pt)

            # Publish Kp estimate and covariance each step
            self.publish_kp()

        # Publish full corrected trajectory
        self.pub_cmd.publish(out)

    # --------------- A-map from IMU ---------------
    def make_A_map_from_imu(self) -> Tuple[Dict[int, np.ndarray], List[str]]:
        A_map: Dict[int, np.ndarray] = {}
        valid_links: List[str] = []
        for link in self.obs_links:
            g_link = self.imu_buf.get_link_g(link)
            if g_link is None:
                continue
            A_f = BinghamUtils.simple_bingham_unit(self.g_base, g_link, parameter=self.parameter_A)
            try:
                fid = self.robot.get_frame_id(link) if self.robot is not None else None
            except Exception:
                fid = None
            if fid is not None:
                A_map[fid] = A_f
                valid_links.append(link)
        return A_map, valid_links

    # --------------- Kp publishers ---------------
    def publish_kp(self) -> None:
        if self.wekf is None:
            return
        kp = np.exp(self.wekf.x)
        P = self.wekf.P

        msg_kp = Float64MultiArray()
        msg_kp.data = kp.tolist()

        msg_cov = Float64MultiArray()
        msg_cov.layout = layout_for_matrix(P.shape[0], P.shape[1])
        msg_cov.data = P.flatten(order="C").tolist()

        self.pub_kp.publish(msg_kp)
        self.pub_cov.publish(msg_cov)

    # --------------- joint indexing utilities ---------------
    def index_of_joint(self, name: str) -> int:
        # position index in robot's actuated joint order
        try:
            return self.actuated_joint_names.index(name)
        except ValueError:
            # fallback: assume same order as incoming names
            return list(self.actuated_joint_names).index(name)

    def extract_theta_from_point(self, names: List[str], pt: JointTrajectoryPoint) -> np.ndarray:
        pos_map = {n: v for n, v in zip(names, pt.positions)}
        theta = np.array([pos_map[n] for n in self.actuated_joint_names], dtype=float)
        return theta

# --------------------- main ---------------------
def main() -> None:
    rospy.init_node("weird_ekf_ros_node")
    node = WeirdEkfRosNode()
    rospy.loginfo("WeirdEKF ROS node started.")
    rospy.spin()

if __name__ == "__main__":
    main()
