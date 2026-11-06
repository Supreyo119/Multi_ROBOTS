
## Multi-Robot Navigation in ROS 2 and Gazebo





A multi-agent robotic simulation framework for **autonomous navigation, formation control, obstacle avoidance, and reinforcement learning**, developed using ROS 2 Jazzy, Gazebo Harmonic, and Python.

The system consists of three mobile robots: one leader and two followers. The leader navigates through an obstacle-filled environment while the follower robots coordinate their movements to maintain formation.

## Simulation Demonstration

![Multi-Robot Gazebo Simulation](multirobots.gif)

## Project Overview

This project explores multi-agent robot navigation by combining traditional path-planning algorithms with learning-based control.

The simulation includes:

- **Three differential-drive robots:** One leader and two followers operating in a shared environment.
- **Autonomous navigation:** A* path planning for the leader with obstacle-aware movement.
- **Formation control:** Two follower robots maintain target positions relative to the leader.
- **Sensor integration:** A 360-degree LiDAR and RGB camera mounted on the leader.
- **ROS 2 communication:** Robot state sharing and coordinated control through ROS 2 topics.
- **Reinforcement learning:** ANN/Transformer-based Deep Q-Network (DQN) for online follower control.
- **Safety mechanisms:** Obstacle-clearance checks, stuck detection, and simulation-based recovery.

## System Architecture

The simulation follows a leader–follower architecture.

**Leader Robot**

The leader uses A* path planning and sensor information to navigate through the environment. It communicates its position, trajectory, and perception information to the followers.

**Follower Robots**

Two follower robots track their desired positions behind the leader while responding to obstacles and changes in formation.

Their controllers combine an artificial neural network (ANN), a Transformer-based observation encoder, and Deep Q-Learning to select control priorities.

The learning controller operates alongside conventional safety and recovery mechanisms.

## Technologies Used

| Component | Technology |
|---|---|
| Robot middleware | ROS 2 Jazzy |
| Simulation | Gazebo Harmonic |
| Programming | Python |
| Path planning | A* |
| Learning | ANN, Transformer, Double DQN |
| Robot communication | ROS 2 / DDS |
| Sensors | LiDAR, RGB camera, odometry |
| Deep learning framework | PyTorch |

## How to Run

### 1. Prerequisites

The Gazebo simulation requires:

- Ubuntu with ROS 2 Jazzy
- Gazebo Harmonic
- `ros_gz_bridge`
- Python virtual environment (`.venv`) with PyTorch

Install and configure the required ROS 2 and Gazebo packages before launching.

### 2. Clone the Repository

```bash
git clone https://github.com/Supreyo119/Multi_ROBOTS.git
cd Multi_ROBOTS
```

### 3. Launch the Gazebo Simulation

After configuring the Python environment and dependencies, run:

```bash
bash Multi_ROBOTS/launch/launch_gazebo.sh
```

This launches the simulation environment, creates three robots, initializes ROS 2 communication, and starts the multi-agent formation controller.

Close Gazebo or press `Ctrl+C` in the launching terminal to stop the simulation.

## Standalone Python Simulation

The project also includes a standalone formation-control simulation using tabular Q-learning.

**Run the simulation:**

```bash
python3 Multi_ROBOTS/scripts/robot_formation.py
```

**Train the controller without the graphical interface:**

```bash
python3 Multi_ROBOTS/scripts/robot_formation.py --headless --episodes 2500
```

**Load a saved policy:**

```bash
python3 Multi_ROBOTS/scripts/robot_formation.py --load
```

The standalone simulation uses tabular Q-learning, whereas the Gazebo implementation uses an ANN/Transformer-based DQN controller.

## Project Structure

```text
Multi_ROBOTS/
├── scripts/       # Navigation and robot control
├── launch/        # Gazebo launcher
├── worlds/        # Gazebo simulation environment
├── config/        # Policies and configuration
├── maps/          # Arena and obstacle maps
├── models/        # Neural network checkpoints
└── tests/         # Simulation and control tests
```

## Future Development

Future directions include:

- Improving multi-agent coordination and formation stability.
- Investigating reinforcement learning for adaptive navigation.
- Evaluating navigation performance under different obstacle configurations.
- Exploring uncertainty-aware path planning and control.
- Transferring simulation-developed controllers to physical mobile robots.

## Additional Documentation

For technical details about the Gazebo architecture, control protocol, and implementation limitations, see [GAZEBO.md](GAZEBO.md).

