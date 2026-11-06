"""Online Transformer DQN selects weights for four motion experts."""
from collections import deque
import random
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(1)
# Each row weights route tracking, formation tracking, recovery, and stopping.
WEIGHTS = [[1, 0, 0, 0], [.8, .2, 0, 0], [.5, .5, 0, 0],
           [.2, .8, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.ann = nn.Sequential(nn.Linear(12, 32), nn.ReLU())
        self.position = nn.Parameter(torch.randn(1, 6, 32)*.02)
        layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(32, len(WEIGHTS))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = .5

    def forward(self, x):
        return self.head(self.transformer(self.ann(x)+self.position)[:, -1])


class Mixer:
    def __init__(self, name):
        torch.manual_seed(7)
        self.net, self.target = Network(), Network()
        self.path = Path(__file__).resolve().parents[1] / 'models' / (name+'_transformer.pt')
        if self.path.exists():
            self.net.load_state_dict(torch.load(self.path, weights_only=True))
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0002)
        self.history = deque(maxlen=6)
        self.replay = deque(maxlen=6000)
        self.previous = None
        self.updates = 0
        self.last_loss = 0.

    def step(self, features, reward, terminal=False):
        self.history.append(features)
        state = [self.history[0]]*(6-len(self.history)) + list(self.history)
        if self.previous:
            old, action = self.previous
            self.replay.append((old, action, reward, state, terminal))
        if len(self.replay) >= 32 and self.updates % 4 == 0:
            batch = random.sample(list(self.replay), 32)
            s, a, r, ns, done = zip(*batch)
            q = self.net(torch.tensor(s, dtype=torch.float32)).gather(1, torch.tensor(a)[:, None]).squeeze(1)
            with torch.no_grad():
                next_action = self.net(torch.tensor(ns, dtype=torch.float32)).argmax(1)
                future = self.target(torch.tensor(ns, dtype=torch.float32)).gather(1, next_action[:, None]).squeeze(1)
                target = torch.tensor(r) + .95 * future * (1-torch.tensor(done, dtype=torch.float32))
            loss = nn.functional.smooth_l1_loss(q, target)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            self.opt.step()
            self.last_loss = loss.item()
        self.updates += 1
        if self.updates % 100 == 0:
            self.target.load_state_dict(self.net.state_dict())
            self.save()
        with torch.no_grad():
            action = self.net(torch.tensor([state], dtype=torch.float32)).argmax(1).item()
        if random.random() < .04:
            action = random.randrange(len(WEIGHTS))
        self.previous = None if terminal else (state, action)
        return WEIGHTS[action]

    def save(self):
        torch.save(self.net.state_dict(), self.path)
