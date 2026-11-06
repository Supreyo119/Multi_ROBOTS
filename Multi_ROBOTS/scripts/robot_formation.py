"""Three differential-drive robots, with shared tabular Q-learning followers."""
import argparse
import json
import math
import random
from pathlib import Path

DT = 0.1
ACTIONS = [(v, w) for v in (0.0, 0.6, 1.2) for w in (-1.8, 0.0, 1.8)]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    def __init__(self, x, y, heading=0):
        self.x, self.y, self.heading = x, y, heading

    def step(self, v, w):
        self.heading = wrap(self.heading + w * DT)
        self.x += v * math.cos(self.heading) * DT
        self.y += v * math.sin(self.heading) * DT


class Policy:
    def __init__(self, seed=7):
        self.q = {}
        self.rng = random.Random(seed)

    def values(self, state):
        return self.q.setdefault(state, [0.0] * len(ACTIONS))

    def act(self, state, epsilon=0):
        if self.rng.random() < epsilon:
            return self.rng.randrange(len(ACTIONS))
        values = self.values(state)
        best = max(values)
        return self.rng.choice([i for i, q in enumerate(values) if q == best])

    def update(self, state, action, reward, next_state, terminal=False):
        values = self.values(state)
        target = reward + (0 if terminal else 0.96 * max(self.values(next_state)))
        values[action] += 0.15 * (target - values[action])

    def save(self, path):
        Path(path).write_text(json.dumps({','.join(map(str, k)): v for k, v in self.q.items()}))

    def load(self, path):
        self.q = {tuple(map(int, k.split(','))): v for k, v in json.loads(Path(path).read_text()).items()}


class World:
    def __init__(self, rng, randomized=False):
        self.time = 0
        self.leader = Robot(0, 0)
        self.followers = [Robot(-1.5, side * 0.9) for side in (1, -1)]
        if randomized:
            for r in self.followers:
                r.x += rng.uniform(-1, 1)
                r.y += rng.uniform(-1, 1)
                r.heading = rng.uniform(-math.pi, math.pi)

    def target(self, index):
        h = self.leader.heading
        side = 0.9 if index == 0 else -0.9
        return (self.leader.x - 1.5 * math.cos(h) - side * math.sin(h),
                self.leader.y - 1.5 * math.sin(h) + side * math.cos(h))

    def observe(self, index):
        r = self.followers[index]
        tx, ty = self.target(index)
        distance = math.hypot(tx - r.x, ty - r.y)
        bearing = wrap(math.atan2(ty - r.y, tx - r.x) - r.heading)
        alignment = wrap(self.leader.heading - r.heading)
        other = self.followers[1 - index]
        clearance = min(math.hypot(r.x - b.x, r.y - b.y) for b in (other, self.leader))
        state = (min(15, int(distance / 0.3)),
                 min(15, int((bearing + math.pi) / (2 * math.pi) * 16)),
                 min(7, int((alignment + math.pi) / (2 * math.pi) * 8)),
                 int(clearance < 0.55))
        return state, distance, clearance

    def step(self, policy, epsilon=0, learn=False):
        before = [self.observe(i) for i in range(2)]
        actions = [policy.act(obs[0], epsilon) for obs in before]
        self.leader.step(0.65, 0.22 * math.sin(self.time * 0.18))
        for robot, action in zip(self.followers, actions):
            robot.step(*ACTIONS[action])
        self.time += DT
        after = [self.observe(i) for i in range(2)]
        terminal = any(obs[1] > 8 for obs in after)
        for i, (state, distance, _) in enumerate(before):
            next_state, new_distance, clearance = after[i]
            reward = 4 * (distance - new_distance) - 0.12 * new_distance
            reward += 0.2 if new_distance < 0.35 else 0
            reward -= 3 if clearance < 0.55 else 0
            if learn:
                policy.update(state, actions[i], reward, next_state, terminal)
        return sum(obs[1] for obs in after) / 2, terminal


def train(policy, episodes, steps):
    for episode in range(episodes):
        world = World(policy.rng, randomized=True)
        epsilon = max(0.04, 0.9 * (1 - episode / max(1, episodes)))
        error = 0
        for step in range(steps):
            distance, terminal = world.step(policy, epsilon, learn=True)
            error += distance
            if terminal:
                break
        if (episode + 1) % max(1, episodes // 10) == 0:
            print(f'Episode {episode + 1}/{episodes}: mean formation error {error / (step + 1):.2f} m')


def evaluate(policy, steps):
    world = World(random.Random(123))
    errors, collisions = [], 0
    for _ in range(steps):
        error, terminal = world.step(policy)
        errors.append(error)
        collisions += int(any(world.observe(i)[2] < 0.55 for i in range(2)))
        if terminal:
            break
    return {'mean_formation_error_m': round(sum(errors) / len(errors), 3),
            'close_proximity_steps': collisions, 'steps': len(errors)}


def show(policy):
    import tkinter as tk
    root = tk.Tk()
    root.title('RL formation — leader + two followers')
    canvas = tk.Canvas(root, width=1000, height=720, bg='#101827')
    canvas.pack()
    world = World(random.Random(123))
    paused = [False]
    root.bind('<space>', lambda event: paused.__setitem__(0, not paused[0]))

    def frame():
        if not paused[0]:
            world.step(policy)
        canvas.delete('all')
        def project(x, y):
            return 600 + (x - world.leader.x) * 65, 350 - (y - world.leader.y) * 65
        canvas.create_text(25, 25, anchor='nw', fill='white', font=('Arial', 18),
                           text='ONE LEADER · TWO RL FOLLOWERS')
        canvas.create_text(25, 60, anchor='nw', fill='#a8b9d0',
                           text='Space: pause/resume | Dashed circles: desired formation positions')
        for i in range(2):
            x, y = project(*world.target(i))
            canvas.create_oval(x-18, y-18, x+18, y+18, outline='#687b96', dash=(4, 4))
        for robot, color, name in zip([world.leader] + world.followers,
                                      ['#ffcf57', '#53c7ff', '#ad8cff'],
                                      ['Leader', 'Follower 1', 'Follower 2']):
            x, y = project(robot.x, robot.y)
            canvas.create_oval(x-14, y-14, x+14, y+14, fill=color)
            canvas.create_line(x, y, x+23*math.cos(robot.heading),
                               y-23*math.sin(robot.heading), fill='white', width=3)
            canvas.create_text(x, y+30, text=name, fill=color)
        error = sum(world.observe(i)[1] for i in range(2)) / 2
        canvas.create_text(25, 680, anchor='nw', fill='white',
                           text=f'Time {world.time:.1f} s    Formation error {error:.2f} m')
        root.after(40, frame)
    frame()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=2500)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--policy', default=str(Path(__file__).resolve().parents[1] / 'config' / 'policy.json'))
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error('episodes and steps must be positive')
    policy = Policy()
    if args.load:
        policy.load(args.policy)
    else:
        train(policy, args.episodes, args.steps)
        policy.save(args.policy)
    print(json.dumps(evaluate(policy, args.steps), indent=2))
    if not args.headless:
        show(policy)


if __name__ == '__main__':
    main()
