import math
import unittest
import xml.etree.ElementTree as ET
import torch
from arena import plan, clearance, WAYPOINTS
from gazebo_world import generate
from neural_mixer import Mixer


class SystemTests(unittest.TestCase):
    def test_only_primary_has_remote_sensors_and_is_larger(self):
        root = ET.parse(generate()).getroot()
        models = {m.attrib['name']: m for m in root.findall('world/model')}
        leader = models['leader']
        self.assertEqual({s.attrib['type'] for s in leader.findall('.//sensor')}, {'camera', 'gpu_lidar', 'contact'})
        for name in ('follower_1', 'follower_2'):
            self.assertEqual({s.attrib['type'] for s in models[name].findall('.//sensor')}, {'contact'})
        self.assertGreater(float(leader.find('link/visual/geometry/box/size').text.split()[0]), .8)

    def test_primary_patrol_routes_clear_obstacles(self):
        start = (0, 0)
        for goal in WAYPOINTS:
            route = plan(start, goal, .8)
            self.assertTrue(route, (start, goal))
            self.assertTrue(all(clearance(*p) >= .8 for p in route))
            start = goal

    def test_transformer_learns_from_negative_transitions(self):
        mixer = Mixer('test_unsaved')
        # Avoid writing test checkpoints; train enough to populate replay and backpropagate.
        before = mixer.net.ann[0].weight.detach().clone()
        for i in range(48):
            weights = mixer.step([.2, 0, 1, .5, float(i % 4 == 0), 0, 0, 0, 0, .5, 0, 0], -8. if i % 4 == 0 else .1)
            self.assertAlmostEqual(sum(weights), 1.)
        self.assertTrue(math.isfinite(mixer.last_loss))
        self.assertFalse(torch.equal(before, mixer.net.ann[0].weight.detach()))
        self.assertTrue(any(t[2] == -8 for t in mixer.replay))


if __name__ == '__main__':
    unittest.main()
