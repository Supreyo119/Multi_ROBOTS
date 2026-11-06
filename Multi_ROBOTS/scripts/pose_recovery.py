"""Position watchdog and collision-checked simulation pose recovery."""
from collections import deque
import math
from arena import plan, clearance


class PositionWatchdog:
    def __init__(self, seconds=3., displacement=.08):
        self.seconds, self.displacement = seconds, displacement
        self.samples = deque()
        self.last_reset = -100.

    def update(self, stamp, pose, needs_motion):
        if self.samples and stamp < self.samples[-1][0]:
            self.samples.clear()
            self.last_reset = -100.
        self.samples.append((stamp, tuple(pose[:2])))
        while len(self.samples) > 1 and self.samples[1][0] <= stamp-self.seconds:
            self.samples.popleft()
        return (needs_motion and stamp-self.last_reset > 8 and self.samples and
                stamp-self.samples[0][0] >= self.seconds and
                max(math.dist(p, pose[:2]) for _, p in self.samples) < self.displacement)

    def reset(self, stamp):
        self.last_reset = stamp
        self.samples.clear()


def recovery_pose(pose, target, points=(), peers=()):
    route = plan(pose[:2], target, .5, points)
    choices = list(reversed(route[:6]))
    choices += [(pose[0]+.75*math.cos(a*math.pi/8), pose[1]+.75*math.sin(a*math.pi/8)) for a in range(16)]
    for x, y in choices:
        if (.3 < math.dist((x,y), pose[:2]) < 1.5 and clearance(x,y) > .48 and
                all(math.dist((x,y), p) > .48 for p in points) and
                all(math.dist((x,y), p[:2]) > .45+p[2] for p in peers)):
            yaw = math.atan2(target[1]-y, target[0]-x)
            return [x, y, .18, 0., 0., yaw]
    return None
