import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from ament_index_python.packages import get_package_share_directory
import os
import subprocess
import tempfile
import yaml
import time


def run_xacro(xacro_path, mappings=""):
    cmd = f"xacro {xacro_path} {mappings}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"xacro failed: {result.stderr}")
    return result.stdout


def load_yaml(file_path):
    """Read a plain (unwrapped) yaml config file as a dict."""
    try:
        with open(file_path) as f:
            return yaml.safe_load(f)
    except OSError:
        return None


class MoveItCommanderNode(Node):
    def __init__(self):
        super().__init__('moveit_commander_node')

        moveit_pkg_share = get_package_share_directory('ur_moveit_config')
        pkg_share = get_package_share_directory('paint_defect_robot')

        # ── Generate robot_description and SRDF as strings ─
        urdf_xacro_path = os.path.join(
            pkg_share, 'urdf', 'ur5e_with_camera.urdf.xacro')
        robot_description = run_xacro(urdf_xacro_path)

        srdf_xacro_path = os.path.join(
            moveit_pkg_share, 'srdf', 'ur.srdf.xacro')
        robot_description_semantic = run_xacro(
            srdf_xacro_path, mappings="name:=ur5e")

        # ── Load the ORIGINAL upstream yaml files as plain
        # dicts (no ROS wrapper) — exactly how move_group's
        # own launch file reads them — then merge everything
        # into ONE dict and wrap it in the ros__parameters
        # envelope ourselves, exactly once, at the end.
        kinematics_yaml = load_yaml(os.path.join(
            moveit_pkg_share, 'config', 'kinematics.yaml')) or {}
        ompl_yaml = load_yaml(os.path.join(
            moveit_pkg_share, 'config', 'ompl_planning.yaml')) or {}
        joint_limits_yaml = load_yaml(os.path.join(
            moveit_pkg_share, 'config', 'joint_limits.yaml')) or {}

        merged_params = {
            'robot_description': robot_description,
            'robot_description_semantic': robot_description_semantic,
            'robot_description_kinematics': kinematics_yaml,
            'robot_description_planning': joint_limits_yaml,
            'planning_pipelines': ['ompl'],
            'default_planning_pipeline': 'ompl',
            'ompl': ompl_yaml,
        }

        wrapped = {
            '/**': {
                'ros__parameters': merged_params
            }
        }

        tmp_yaml = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False)
        yaml.dump(wrapped, tmp_yaml, default_flow_style=False)
        tmp_yaml.close()

        self.get_logger().info(f'Using combined param file: {tmp_yaml.name}')

        # ── MoveIt setup ──────────────────────────────────
        self.moveit = MoveItPy(
            node_name='moveit_py_commander',
            launch_params_filepaths=[tmp_yaml.name]
        )
        self.arm = self.moveit.get_planning_component('ur_manipulator')
        self.robot = self.moveit.get_robot_model()

        # ── Poses in radians ──────────────────────────────
        self.inspect_pose = [0.0, -1.29, 1.83, -0.61, 1.50, 0.0]
        self.pickup_pose  = [0.0, -1.13, 1.60, -0.52, 1.50, 0.0]
        self.drop_pose    = [3.10, -1.34, 1.93, -0.62, 1.50, 0.0]

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # ── State ─────────────────────────────────────────
        self.is_moving = False
        self.waiting_for_inspection = True

        # ── Subscriber ────────────────────────────────────
        self.sub = self.create_subscription(
            String,
            '/defect_detection/result',
            self.result_callback,
            10)

        self.get_logger().info('MoveIt commander node ready')
        self.get_logger().info('Waiting for defect detection results...')

    def move_to_joint_angles(self, joint_angles, pose_name):
        """
        Move arm to given joint angles.
        Returns True if successful, False otherwise.
        """
        self.get_logger().info(f'Moving to {pose_name}...')

        # Set start state to current
        self.arm.set_start_state_to_current_state()

        # Build robot state with target joint angles
        robot_state = RobotState(self.robot)
        robot_state.set_joint_group_positions(
            'ur_manipulator',
            joint_angles)

        # Set goal
        self.arm.set_goal_state(robot_state=robot_state)

        # Plan
        plan_result = self.arm.plan()

        if plan_result:
            self.get_logger().info(f'Plan succeeded for {pose_name} — executing')
            self.moveit.execute(plan_result.trajectory, controllers=[])
            self.get_logger().info(f'Reached {pose_name}')
            return True
        else:
            self.get_logger().error(f'Planning failed for {pose_name}')
            return False

    def handle_defect(self):
        """
        Full pick and place sequence when defect is detected.
        """
        self.get_logger().warn('DEFECT DETECTED — starting pick and place sequence')
        self.is_moving = True

        try:
            # Step 1 — move to pickup pose
            success = self.move_to_joint_angles(
                self.pickup_pose, 'pickup_pose')
            if not success:
                return

            # Step 2 — simulate gripper close
            self.get_logger().info('Gripper CLOSE (simulated)')
            time.sleep(1.0)

            # Step 3 — move to drop pose
            success = self.move_to_joint_angles(
                self.drop_pose, 'drop_pose')
            if not success:
                return

            # Step 4 — simulate gripper open
            self.get_logger().info('Gripper OPEN (simulated)')
            time.sleep(1.0)

            # Step 5 — return to inspect pose
            success = self.move_to_joint_angles(
                self.inspect_pose, 'inspect_pose')

            self.get_logger().info('Pick and place complete — resuming inspection')

        except Exception as e:
            self.get_logger().error(f'Error during pick and place: {e}')
        finally:
            self.is_moving = False

    def result_callback(self, msg):
        """
        Called every time CNN publishes a result.
        """
        # Ignore messages while arm is moving
        if self.is_moving:
            return

        result = msg.data

        if 'DEFECT' in result:
            self.handle_defect()
        elif 'OK' in result:
            self.get_logger().info(f'Surface OK — no action needed')
        else:
            self.get_logger().info(f'Uncertain — ignoring')


def main(args=None):
    rclpy.init(args=args)
    node = MoveItCommanderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
