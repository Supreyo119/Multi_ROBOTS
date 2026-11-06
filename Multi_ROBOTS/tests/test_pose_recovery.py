import math
import unittest
from pose_recovery import PositionWatchdog, recovery_pose
from arena import clearance


class RecoveryTests(unittest.TestCase):
    def test_stationary_robot_is_detected_without_velocity_command(self):
        watch = PositionWatchdog()
        for i in range(30):
            self.assertFalse(watch.update(i/10, [0,0,0], True))
        self.assertTrue(watch.update(3., [0,0,0], True))
        watch.reset(3.)
        self.assertFalse(watch.update(6., [0,0,0], True))

    def test_formation_stop_and_moving_robot_do_not_trigger(self):
        watch = PositionWatchdog()
        for i in range(60):
            self.assertFalse(watch.update(i/10, [0,0,0], False))
        watch = PositionWatchdog()
        for i in range(60):
            self.assertFalse(watch.update(i/10, [i/20,0,0], True))

    def test_recovery_order_is_clear_and_has_rpy(self):
        pose = [-1.5,-.9,0]
        order = recovery_pose(pose, [0,-2], peers=[(0,0,.55)])
        self.assertIsNotNone(order)
        x,y,z,r,p,yaw = order
        self.assertGreater(clearance(x,y), .48)
        self.assertGreater(math.dist([x,y],[0,0]), 1.)
        self.assertEqual([r,p],[0,0])
        self.assertTrue(0 < math.dist(pose[:2], order[:2]) < 1.5)


if __name__ == '__main__':
    unittest.main()
