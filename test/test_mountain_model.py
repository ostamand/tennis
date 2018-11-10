import unittest
from mountain.model import Actor
from mountain.model import Critic
import gym
import torch 
import pdb

class TestGymModel(unittest.TestCase):

    def tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)

    def setUp(self):
        self.env = gym.make('MountainCarContinuous-v0')
        self.state = self.env.reset()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def test_can_create_actor(self):
        actor = Actor(self.state_size, self.action_size)
        action = actor(self.tensor(self.state)).item()
        self.assertGreaterEqual(action, -1)
        self.assertLessEqual(action, 1)

    def test_actor_seed(self):
        actor1 = Actor(self.state_size, self.action_size, seed=42)
        actor2 = Actor(self.state_size, self.action_size, seed=42)
        actor3 = Actor(self.state_size, self.action_size, seed=12)
        self.assertAlmostEqual(actor1.fc1.weight[0, 0], actor2.fc1.weight[0, 0])
        self.assertNotEqual(actor1.fc1.weight[0, 0], actor3.fc1.weight[0, 0])

    def test_can_create_critic(self):
        critic = Critic(self.state_size, self.action_size)
        actor = Actor(self.state_size, self.action_size)

        state = self.tensor(self.state)
        action = actor(state)
        q = critic(state.unsqueeze(0), action.unsqueeze(0))
        self.assertIsNotNone(q)
