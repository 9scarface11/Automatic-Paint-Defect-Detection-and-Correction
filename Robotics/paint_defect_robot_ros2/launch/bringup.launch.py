import os
import yaml
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # ─────────────────────────────────────────
    # ARGUMENTS
    # ─────────────────────────────────────────
    ur_type_arg = DeclareLaunchArgument(
        'ur_type',
        default_value='ur5e',
        description='Type of UR robot'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # ─────────────────────────────────────────
    # PACKAGE PATH
    # ─────────────────────────────────────────
    pkg_share = FindPackageShare('paint_defect_robot')

    # ─────────────────────────────────────────
    # URDF from xacro
    # Converts our xacro template into a URDF string.
    # robot_state_publisher reads this and publishes
    # /robot_description and /tf topics.
    # ─────────────────────────────────────────
    robot_description = ParameterValue(
        Command([
            FindExecutable(name='xacro'), ' ',
            PathJoinSubstitution([
                pkg_share, 'urdf', 'ur5e_with_camera.urdf.xacro'
            ])
        ]),
        value_type=str
    )

    # ─────────────────────────────────────────
    # ROBOT STATE PUBLISHER
    # Publishes /robot_description and /tf tree.
    # Every other node depends on this.
    # ─────────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }]
    )

    # ─────────────────────────────────────────
    # GAZEBO
    # Starts Gz Sim with an empty world.
    # GZ_SIM_SYSTEM_PLUGIN_PATH tells Gazebo where
    # to find gz_ros2_control plugin.
    # GZ_SIM_RESOURCE_PATH tells Gazebo where to
    # find UR meshes.
    # ─────────────────────────────────────────
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', 'empty.sdf'],
        output='screen',
        additional_env={
            'GZ_SIM_SYSTEM_PLUGIN_PATH': '/opt/ros/jazzy/lib',
            'GZ_SIM_RESOURCE_PATH': os.path.join(
                get_package_share_directory('ur_description'), '..'
            )
        }
    )

    # ─────────────────────────────────────────
    # SPAWN ROBOT
    # Reads /robot_description topic and spawns
    # the robot model inside Gazebo.
    # ─────────────────────────────────────────
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'ur5e',
            '-topic', '/robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    # ─────────────────────────────────────────
    # BRIDGE
    # Connects Gazebo topics to ROS2 topics.
    # /clock      → simulation time
    # /camera/*   → camera images and info
    # ─────────────────────────────────────────
    bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
    ],
    remappings=[
        ('/camera/image_raw', '/camera/image_raw'),
    ],
    output='screen'
)

    # ─────────────────────────────────────────
    # CONTROLLERS
    # joint_state_broadcaster → reads joint angles
    #   from Gazebo, publishes to /joint_states
    # joint_trajectory_controller → accepts motion
    #   commands and drives the joints
    # ─────────────────────────────────────────
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager'
        ],
        output='screen'
    )

    trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_trajectory_controller',
            '--controller-manager', '/controller_manager'
        ],
        output='screen'
    )

    # ─────────────────────────────────────────
    # MOVEIT CONFIG
    # MoveItConfigsBuilder reads ur_moveit_config
    # package for kinematics, OMPL, joint limits etc.
    # We then inject our own controllers config on
    # top so MoveIt uses joint_trajectory_controller
    # instead of scaled_joint_trajectory_controller.
    # ─────────────────────────────────────────
    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description_semantic(Path("srdf") / "ur.srdf.xacro", {"name": "ur5e"})
        .to_moveit_configs()
    )

    # Our controllers config — overrides the default
    # which points to scaled_joint_trajectory_controller
    moveit_controllers = {
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/MoveItSimpleControllerManager"
        ),
        "trajectory_execution": {
            "allowed_execution_duration_scaling": 1.2,
            "allowed_goal_duration_margin": 0.5,
            "allowed_start_tolerance": 0.01,
            "execution_duration_monitoring": False,
        },
        "moveit_simple_controller_manager": {
            "controller_names": ["joint_trajectory_controller"],
            "joint_trajectory_controller": {
                "action_ns": "follow_joint_trajectory",
                "type": "FollowJointTrajectory",
                "default": True,
                "joints": [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
            },
        },
    }

    warehouse_ros_config = {
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "warehouse_host": os.path.expanduser("~/.ros/warehouse_ros.sqlite"),
    }

    # ─────────────────────────────────────────
    # MOVE GROUP NODE
    # The core MoveIt node that handles:
    # - Motion planning (OMPL)
    # - Collision checking
    # - Trajectory execution
    # We pass our moveit_controllers dict here
    # which overrides the default controller config.
    # ─────────────────────────────────────────
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            moveit_controllers,
            warehouse_ros_config,
            {"use_sim_time": True},
        ],
    )

    # ─────────────────────────────────────────
    # RVIZ
    # Visualization + Motion Planning panel.
    # Loads MoveIt RViz config from ur_moveit_config.
    # ─────────────────────────────────────────
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("ur_moveit_config"), "config", "moveit.rviz"
    ])

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            warehouse_ros_config,
            {"use_sim_time": True},
        ],
    )

    # ─────────────────────────────────────────
    # MOVEIT TIMER
    # Delay MoveIt by 5 seconds to make sure:
    # - Gazebo is running
    # - Robot is spawned
    # - Controllers are active
    # - /robot_description is being published
    # ─────────────────────────────────────────
    moveit = TimerAction(
        period=5.0,
        actions=[move_group_node, rviz_node]
    )

    # ─────────────────────────────────────────
    # LAUNCH EVERYTHING
    # ─────────────────────────────────────────
    return LaunchDescription([
        ur_type_arg,
        use_sim_time_arg,
        robot_state_publisher,
        gazebo,
        spawn_robot,
        bridge,
        joint_state_broadcaster_spawner,
        trajectory_controller_spawner,
        moveit,
    ])
