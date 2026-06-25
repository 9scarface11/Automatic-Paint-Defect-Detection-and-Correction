# ROS2 Paint Defect Robot — Simulation Pipeline

ROS2 Jazzy robotics pipeline for automated paint defect inspection and correction using a UR5e arm in Gazebo Harmonic.
------------------------------------------------------------------------------------------------------------------
## Pipeline
fake_image_publisher (or real camera)

→ /camera/image_raw

→ cnn_node (ResNet18, PyTorch)

→ /defect_detection/result

→ moveit_commander_node

→ MoveIt2 → joint_trajectory_controller

→ UR5e arm motion in Gazebo
------------------------------------------------------------------------------------------------------------------
## Package Structure
paint_defect_robot_ros2/

├── config/

│   ├── ros2_controllers.yaml     # joint_state_broadcaster + joint_trajectory_controller

│   └── moveit_controllers.yaml   # MoveIt2 controller override (standard JTC)

├── launch/

│   └── bringup.launch.py         # launches Gazebo, MoveIt2, RViz, bridge, controllers

├── urdf/

│   └── ur5e_with_camera.urdf.xacro  # UR5e + camera link + ros2_control + Gazebo sensor

├── models/

│   └── model.pth                 # trained ResNet18 weights (not tracked by git)

└── paint_defect_robot/

├── fake_image_publisher.py   # synthetic camera feed (VirtualBox GPU workaround)

├── cnn_node.py               # PyTorch ResNet18 inference node

└── moveit_commander_node.py  # MoveItPy pick-and-place sequencer
------------------------------------------------------------------------------------------------------------------
## Nodes and Topics

| Node | Subscribes | Publishes |
|---|---|---|
| `fake_image_publisher` | — | `/camera/image_raw` |
| `cnn_node` | `/camera/image_raw` | `/defect_detection/result` |
| `moveit_commander_node` | `/defect_detection/result` | MoveIt2 goal |

------------------------------------------------------------------------------------------------------------------
## Motion Sequence (on DEFECT detection)
inspect_pose → pickup_pose → [gripper close] → drop_pose → [gripper open] → inspect_pose
Gripper is simulated (no physical gripper modeled) — represented as log messages at the intended grasp points.

## Key Technical Details

- `gz_ros2_control` bridges Gazebo physics to `ros2_control` hardware interface
- MoveIt2 uses OMPL (RRTConnect) for motion planning
- Camera rendering blocked in VirtualBox (no GPU passthrough) — `fake_image_publisher` is the workaround; swap for any real `sensor_msgs/Image` publisher on physical hardware
- `moveit_commander_node` uses `MoveItPy` with manually merged config (robot_description + SRDF + OMPL pipeline) due to `MoveItConfigsBuilder` auto-discovery limitations in standalone Python nodes
------------------------------------------------------------------------------------------------------------------
## How to Run

```bash
# Terminal 1 — full simulation stack
ros2 launch paint_defect_robot bringup.launch.py

# Terminal 2 — synthetic camera
ros2 run paint_defect_robot fake_image_publisher

# Terminal 3 — CNN inference
ros2 run paint_defect_robot cnn_node --ros-args \
  -p model_path:=/path/to/model.pth \
  -p confidence_threshold:=0.7

# Terminal 4 — motion commander
ros2 run paint_defect_robot moveit_commander_node
```
------------------------------------------------------------------------------------------------------------------
## Environment

- OS: Ubuntu 24.04
- ROS2: Jazzy
- Gazebo: Harmonic
- Python: 3.12
- PyTorch: 2.12.1 (CPU)
