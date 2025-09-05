# IK with IKPy/pyik using your URDF and a FK-made target
import numpy as np
import pinocchio as pin
import os

# pick ONE:
# import ikpy as ik  # if you installed the GitHub fork 'pyik' (kuanfang/pyik)
from ikpy.chain import Chain as IkChain  # if you installed IKPy from pip

# ensure_urdf() はあなたの定義をそのまま利用
# from your_module import ensure_urdf

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

def solve_ik_with_ikpy():
    urdf_path = ensure_urdf()  # simple6r.urdf を生成
    # IKPy/pyik: URDF からチェーン生成（base_link を起点に自動でたどる）
    chain = IkChain.from_urdf_file(
        urdf_path,
        base_elements=["base_link"],  # 省略時も "base_link" が既定
        symbolic=False
    )

    # Pinocchio: ランダム関節から目標姿勢（必ず可到達）
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    q_ref = np.array([
        np.random.uniform(lo, hi)
        for lo, hi in zip(model.lowerPositionLimit, model.upperPositionLimit)
    ])
    pin.forwardKinematics(model, data, q_ref)
    joint_id = model.getJointId("joint6")
    oM_des = pin.SE3(data.oMi[joint_id])  # 参照切り離し
    T_target = se3_to_homog(oM_des)

    # 初期値（全ゼロでOK。必要なら別の初期値を与える）
    # IK を "フレーム指定（4x4）" で解く：位置＋姿勢を同時に最適化
    q0_full = np.zeros(len(chain.links))
    q_sol_full = chain.inverse_kinematics_frame(
        T_target,
        initial_position=q0_full,
        max_iter=1000,                 # ikpy >=3.3 の引数
        orientation_mode="all"         # 姿勢も最適化（内部で使われます）
    )

    # IKPy の戻りは「OriginLink 含むフル列」。実関節は先頭を除いた部分
    q_sol = np.array(q_sol_full[1:1+model.nq])  # joint1..6 を抽出

    # 検算（IKPyの順運動学）
    T_final = chain.forward_kinematics(q_sol_full)

    # 誤差表示
    R_t = T_target[:3, :3]; t_t = T_target[:3, 3]
    R_f = T_final[:3, :3];  t_f = T_final[:3, 3]
    R_err = R_t.T @ R_f
    # ねじれ角（0..pi）へ丸め
    cosang = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    ang_err = float(np.arccos(cosang))
    lin_err = float(np.linalg.norm(t_f - t_t))

    print("=== IKPy result ===")
    print(f"q_ref (pin): {q_ref.tolist()}")
    print(f"q_sol (ikpy): {q_sol.tolist()}")
    print(f"linear error [m]: {lin_err:.3e}")
    print(f"angular error [rad]: {ang_err:.3e}")

if __name__ == "__main__":
    solve_ik_with_ikpy()
