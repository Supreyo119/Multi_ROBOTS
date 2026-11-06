"""Independent ROS nodes: primary perception and communicated follower RL."""
from collections import deque
import json
import math
import time
import subprocess
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String
from ros_gz_interfaces.msg import Contacts
from arena import plan, clearance, WAYPOINTS
from neural_mixer import Mixer
from pose_recovery import PositionWatchdog, recovery_pose
from robot_formation import wrap

QOS = QoSProfile(depth=3, reliability=ReliabilityPolicy.RELIABLE)
OFFSETS = {'leader': (0, 0), 'follower_1': (-1.5, .9), 'follower_2': (-1.5, -.9)}


def steer(pose, target, speed=.65):
    dx, dy = target[0]-pose[0], target[1]-pose[1]
    angle = wrap(math.atan2(dy, dx)-pose[2])
    return min(speed, math.hypot(dx, dy))*max(0., math.cos(angle))**2, max(-1.3, min(1.3, 2.4*angle))


def route_command(pose, route, speed):
    if not route:
        return 0., 0.
    nearest = min(range(len(route)), key=lambda i: math.dist(pose[:2], route[i]))
    return steer(pose, route[min(nearest+2, len(route)-1)], speed)


class RobotNode(Node):
    def __init__(self, name):
        super().__init__(name+'_controller')
        self.name = name
        self.pose = [*OFFSETS[name], 0.]
        self.last_odom = self.sim_time = self.contact_until = 0.
        self.command = (0., 0.)
        self.pub = self.create_publisher(Twist, f'/model/{name}/cmd_vel', 10)
        self.sub = self.create_subscription(Odometry, f'/model/{name}/odometry', self.odom, qos_profile_sensor_data)
        self.bump = self.create_subscription(Contacts, f'/model/{name}/contacts', self.contact, qos_profile_sensor_data)

    def odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = [p.x, p.y,
                     math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))]
        self.last_odom = time.monotonic()
        self.sim_time = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9

    def contact(self, msg):
        if msg.contacts:
            self.contact_until = self.sim_time+.4

    def send(self, v, w):
        if not rclpy.ok():
            return
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.pub.publish(msg)
        self.command = v, w

    def shield(self, command, radius, points=(), peers=()):
        v, w = command
        for t in (.2, .4, .7):
            h = self.pose[2]+w*t/2
            x, y = self.pose[0]+v*t*math.cos(h), self.pose[1]+v*t*math.sin(h)
            if clearance(x, y) < radius or any(math.hypot(x-p[0], y-p[1]) < radius for p in points) or any(math.hypot(x-p[0], y-p[1]) < radius+p[2] for p in peers):
                return 0., w
        return v, w


class Primary(RobotNode):
    def __init__(self):
        super().__init__('leader')
        self.tx = self.create_publisher(String, '/formation/primary', QOS)
        self.scan_sub = self.create_subscription(LaserScan, '/leader/scan', self.scan, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(Image, '/leader/camera', self.image, qos_profile_sensor_data)
        self.status_sub = self.create_subscription(String, '/formation/feedback', self.feedback, QOS)
        self.points, self.vision = [], [0., 0., 0.]
        self.scan_time = self.image_time = 0.
        self.scan_count = self.image_count = 0
        self.followers = {}
        self.trail = deque(maxlen=180)
        self.route = []
        self.waypoint = self.seq = 0
        self.replan = self.report = 0.
        self.create_timer(.1, self.control)

    def scan(self, msg):
        self.scan_time = time.monotonic()
        self.scan_count += 1
        points = []
        for i in range(0, len(msg.ranges), 3):
            r = msg.ranges[i]
            if math.isfinite(r) and .3 < r < min(msg.range_max, 10.):
                h = self.pose[2]+msg.angle_min+i*msg.angle_increment
                points.append([round(self.pose[0]+r*math.cos(h), 2), round(self.pose[1]+r*math.sin(h), 2)])
        self.points = points

    def image(self, msg):
        self.image_time = time.monotonic()
        self.image_count += 1
        # Three image-sector red-obstacle cues, not metric depth or object recognition.
        if msg.encoding.lower() not in ('rgb8', 'bgr8'):
            return
        values, counts = [0]*3, [0]*3
        for y in range(msg.height//3, msg.height*3//4, 8):
            for x in range(0, msg.width, 8):
                idx = y*msg.step+x*3
                rgb = msg.data[idx:idx+3]
                if len(rgb) < 3:
                    continue
                r, g, b = rgb if msg.encoding.lower() == 'rgb8' else rgb[::-1]
                sector = min(2, x*3//msg.width)
                values[sector] += int(r > 1.5*g and r > 1.5*b and r > 60)
                counts[sector] += 1
        self.vision = [a/max(1, b) for a, b in zip(values, counts)]

    def feedback(self, msg):
        try:
            data = json.loads(msg.data)
            if data.get('protocol') == 'formation/1' and data.get('robot') in ('follower_1', 'follower_2'):
                self.followers[data['robot']] = (time.monotonic(), data)
        except (ValueError, KeyError):
            pass

    def control(self):
        now = time.monotonic()
        if not all(now-t < 2 for t in (self.last_odom, self.scan_time, self.image_time)):
            self.send(0, 0)
            return
        if (math.dist(self.pose[:2], WAYPOINTS[self.waypoint]) < .65 or
                (self.route and math.dist(self.pose[:2], self.route[-1]) < .35)):
            self.waypoint = (self.waypoint+1) % len(WAYPOINTS)
            self.route = []
        if not self.route or now-self.replan > 2:
            self.route = plan(self.pose[:2], WAYPOINTS[self.waypoint], .85, self.points)
            self.replan = now
        if not self.trail or math.dist(self.pose[:2], self.trail[-1][:2]) > .12:
            self.trail.append(list(self.pose))
        peers = [(*d['pose'][:2], .35) for t, d in self.followers.values() if now-t < 1]
        lag = max([math.dist(self.pose[:2], p[:2]) for p in peers] or [0])
        command = route_command(self.pose, self.route, .5 if lag < 3 else .2)
        if self.vision[1] > .45:
            command = command[0]*.5, command[1]
        self.send(*self.shield(command, .68, self.points, peers))
        self.seq += 1
        packet = {'protocol': 'formation/1', 'seq': self.seq, 'sim_time': self.sim_time,
                  'ttl': .8, 'pose': self.pose, 'trail': list(self.trail),
                  'points': self.points, 'vision': self.vision,
                  'peers': {n: d['pose'] for n, (t, d) in self.followers.items() if now-t < 1}}
        self.tx.publish(String(data=json.dumps(packet)))
        if now-self.report > 5:
            self.get_logger().info(f'primary pose={self.pose[:2]} lidar={self.scan_count} camera={self.image_count} waypoint={self.waypoint} packets={self.seq}')
            self.report = now


class Follower(RobotNode):
    def __init__(self, name, offset):
        super().__init__(name)
        self.offset = offset
        self.rx = self.create_subscription(String, '/formation/primary', self.packet, QOS)
        self.tx = self.create_publisher(String, '/formation/feedback', QOS)
        self.data = None
        self.rx_time = 0.
        self.seq = -1
        self.mixer = Mixer(name)
        self.route = []
        self.replan = self.report = 0.
        self.previous_distance = None
        self.motion = deque(maxlen=20)
        self.watchdog = PositionWatchdog()
        self.pose_request = None
        self.pose_resets = 0
        self.total_reward = 0.
        self.collisions = self.stuck_steps = 0
        self.create_timer(.1, self.control)

    def packet(self, msg):
        try:
            data = json.loads(msg.data)
            if data['protocol'] != 'formation/1' or data['seq'] <= self.seq:
                return
            if abs(self.sim_time-data['sim_time']) > data['ttl']:
                return
            self.data, self.seq, self.rx_time = data, data['seq'], time.monotonic()
        except (ValueError, KeyError, TypeError):
            return

    def target(self):
        trail = self.data['trail']
        target = self.data['pose']
        length = 0.
        for a, b in zip(reversed(trail[:-1]), reversed(trail[1:])):
            length += math.dist(a[:2], b[:2])
            target = a
            if length >= 1.8:
                break
        if length < 1.8:
            target = [target[0]-(1.8-length)*math.cos(target[2]),
                      target[1]-(1.8-length)*math.sin(target[2]), target[2]]
        return [target[0]-self.offset*math.sin(target[2]), target[1]+self.offset*math.cos(target[2])]

    def control(self):
        now = time.monotonic()
        if not self.data or now-self.rx_time > min(.8, self.data['ttl']) or now-self.last_odom > 1:
            self.send(0, 0)
            return
        if self.pose_request is not None:
            self.send(0, 0)
            if self.pose_request.poll() is None:
                return
            output, error = self.pose_request.communicate()
            if self.pose_request.returncode == 0 and 'true' in output:
                self.pose_resets += 1
                self.get_logger().warning(f'XYZ/RPY pose recovery confirmed, count={self.pose_resets}')
            else:
                self.get_logger().error(f'Pose recovery failed: {output} {error}')
            self.pose_request = None
            self.route = []
            self.motion.clear()
            self.previous_distance = None
            self.mixer.previous = None
            self.mixer.history.clear()
            return
        target = self.target()
        distance = math.dist(self.pose[:2], target)
        if not self.route or now-self.replan > .8:
            self.route = plan(self.pose[:2], target, .48, self.data['points'])
            self.replan = now
        route, formation = route_command(self.pose, self.route, .8), steer(self.pose, target, .8)
        if distance < .2:
            route = formation = (0., 0.)
        self.motion.append((self.sim_time, *self.pose[:2], self.command[0]))
        stalled = (len(self.motion) == 20 and
                   self.motion[-1][0]-self.motion[0][0] > 1.5 and
                   math.dist(self.motion[0][1:3], self.pose[:2]) < .08 and distance > .5)
        stuck = (len(self.motion) == 20 and self.motion[-1][0]-self.motion[0][0] > 1 and
                 math.dist(self.motion[0][1:3], self.pose[:2]) < .08 and
                 sum(abs(p[3]) for p in self.motion)/20 > .15 and distance > .4)
        collision = self.sim_time < self.contact_until
        progress = 0 if self.previous_distance is None else max(-.2, min(.2, self.previous_distance-distance))
        reward = 5*progress-.02*distance-8*int(collision)-2*int(stuck or stalled)+(.15 if distance < .3 else 0)
        self.previous_distance = distance
        self.total_reward += reward
        self.collisions += int(collision)
        self.stuck_steps += int(stuck)
        angle = wrap(math.atan2(target[1]-self.pose[1], target[0]-self.pose[0])-self.pose[2])
        features = [min(distance/5, 1), math.sin(angle), math.cos(angle),
                    min(clearance(*self.pose[:2])/3, 1), float(collision), float(stuck),
                    *self.data['vision'], self.command[0], self.command[1]/1.3, min((now-self.rx_time)/.8, 1)]
        weights = self.mixer.step(features, reward, collision)
        recovery = (-.18, .8 if self.offset > 0 else -.8)
        experts = [route, formation, recovery, (0., 0.)]
        command = tuple(sum(w*expert[i] for w, expert in zip(weights, experts)) for i in range(2))
        if stalled and not collision:
            # A learned stop choice must not strand a follower away from formation.
            command = route
        if collision or stuck:
            command = recovery
        peers = [(*self.data['pose'][:2], .55)] + [(*p[:2], .32) for n, p in self.data.get('peers', {}).items() if n != self.name]
        points = [p for p in self.data['points'] if all(math.dist(p, q[:2]) > q[2]+.18 for q in peers)
                  and math.dist(p, self.pose[:2]) > .5]
        if self.watchdog.update(self.sim_time, self.pose, distance > .5):
            order = recovery_pose(self.pose, target, points, peers)
            if order is not None:
                x, y, z, roll, pitch, yaw = order
                # Roll and pitch are zero for the level-ground differential-drive base.
                request = (f'name: "{self.name}" position: {{x: {x} y: {y} z: {z}}} '
                           f'orientation: {{x: 0 y: 0 z: {math.sin(yaw/2)} w: {math.cos(yaw/2)}}}')
                self.send(0, 0)
                self.get_logger().warning(f'POSITION WATCHDOG: stationary for 3s; XYZ/RPY order={order}')
                self.pose_request = subprocess.Popen(
                    ['gz', 'service', '-s', '/world/formation/set_pose', '--reqtype', 'gz.msgs.Pose',
                     '--reptype', 'gz.msgs.Boolean', '--timeout', '2000', '--req', request],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.watchdog.reset(self.sim_time)
                return
        self.send(*self.shield(command, .34, points, peers))
        status = {'protocol': 'formation/1', 'robot': self.name, 'seq': self.seq,
                  'pose': self.pose, 'reward': reward, 'collision': collision, 'stuck': stuck,
                  'weights': weights, 'error': distance, 'loss': self.mixer.last_loss,
                  'stalled': stalled, 'command': list(self.command), 'pose_resets': self.pose_resets}
        self.tx.publish(String(data=json.dumps(status)))
        if now-self.report > 5:
            self.get_logger().info(f'pose={self.pose[:2]} command={self.command} stalled={stalled} error={distance:.2f} reward={reward:.3f} weights={weights} collision_steps={self.collisions} stuck_steps={self.stuck_steps} updates={self.mixer.updates} loss={self.mixer.last_loss:.4f}')
            self.report = now


def main():
    rclpy.init()
    nodes = [Primary(), Follower('follower_1', .75), Follower('follower_2', -.75)]
    executor = SingleThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        for node in nodes:
            if rclpy.ok():
                node.send(0, 0)
            if isinstance(node, Follower):
                node.mixer.save()
            node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
