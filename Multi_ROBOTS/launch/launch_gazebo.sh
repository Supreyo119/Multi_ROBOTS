#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/../.venv/bin/python"
export GZ_PARTITION=rl_three_robot_formation
export ROS_DOMAIN_ID=47
exec 9>.gazebo.lock
flock -n 9 || { echo "This robot simulation is already running"; exit 1; }
/usr/bin/python3 scripts/gazebo_world.py
"$PYTHON" -c 'import torch'
gz sim -r worlds/formation.sdf >gazebo.log 2>&1 &
sim_pid=$!
args=()
for robot in leader follower_1 follower_2; do
  args+=("/model/$robot/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist")
  args+=("/model/$robot/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry")
  args+=("/model/$robot/contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts")
done
args+=("/leader/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan")
args+=("/leader/camera@sensor_msgs/msg/Image[gz.msgs.Image")
/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge "${args[@]}" >bridge.log 2>&1 &
bridge_pid=$!
"$PYTHON" -u scripts/gazebo_control.py >controller.log 2>&1 &
control_pid=$!
cleanup() {
  kill -INT "$control_pid" 2>/dev/null || true
  sleep 0.3
  pkill -TERM -P "$sim_pid" 2>/dev/null || true
  sleep 0.5
  pkill -KILL -P "$sim_pid" 2>/dev/null || true
  kill "$bridge_pid" "$sim_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 0' INT TERM
echo "Gazebo running. Logs: gazebo.log, bridge.log, controller.log"
wait "$sim_pid"
