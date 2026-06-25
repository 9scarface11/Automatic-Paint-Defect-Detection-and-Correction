# Automated Paint Defect Detection and Correction

A full-stack robotics project combining a trained CNN classifier with a ROS2-based robot arm pipeline to automatically detect and remove defective painted surfaces in simulation.
------------------------------------------------------------------------------------------------------------------
## Pipeline
Camera image

→ ResNet18 CNN (defect / ok)

→ ROS2 decision node

→ MoveIt2 motion planning

→ UR5e picks defective part → drops at disposal location → returns to inspect

## Repository Structure
├── CNN/          # ResNet18 training, DAGM 2007 dataset, model weights

├── Robotics/     # ROS2 Jazzy pipeline, Gazebo Harmonic simulation

└── Experimental/ # Early prototypes and experiments
------------------------------------------------------------------------------------------------------------------
## Results

| Metric | Value |
|---|---|
| Validation Accuracy | 91% |
| Defect F1-Score | 0.85 |
| Dataset | DAGM 2007 (binary: defect / ok) |
| Robot | Universal Robots UR5e |
| Simulation | Gazebo Harmonic |

## Stack

- **Perception:** PyTorch, ResNet18, OpenCV, cv_bridge
- **Robot Middleware:** ROS2 Jazzy, ros2_control, gz_ros2_control
- **Motion Planning:** MoveIt2, OMPL (RRTConnect)
- **Simulation:** Gazebo Harmonic, Gz Sim
- **Robot Model:** Universal Robots UR5e

## How They Connect

The `CNN/` folder contains the standalone training pipeline. The trained `model.pth` is loaded at runtime by the ROS2 `cnn_node` in `Robotics/`, which subscribes to the robot's camera topic, runs inference on each frame, and publishes the result. The `moveit_commander_node` receives that result and triggers the appropriate arm motion sequence via MoveIt2.
