import torch
import numpy as np

from ddpg.noise import OUNoise
from ddpg.replay_buffer import ReplayBuffer

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Agent():
    def __init__(self, state_size, action_size, actor, critic,
                 action_low=-1.0, action_high=1.0,
                 lrate=1e-4, tau=0.01, buffer_size=1e5, batch_size=64,
                 exploration_mu=0.0, exploration_theta=0.15,
                 exploration_sigma=0.20,
                 seed=None):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.seed = seed if seed else 0
        self.lrate = lrate
        self.tau = tau
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.device = torch.device(DEVICE)

        # actors networks
        self.actor = actor(state_size, action_size,
                           low=action_low, high=action_high, seed=self.seed)
        self.actor_target = actor(state_size, action_size,
                                  low=action_low, high=action_high, seed=self.seed)

        # critic networks
        self.critic = critic(state_size, action_size, seed=self.seed)
        self.critic_target = critic(state_size, action_size, seed=self.seed)

        # noise 
        self.noise = OUNoise(action_size, exploration_mu, exploration_theta, exploration_sigma)

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.device, self.buffer_size, self.batch_size)

        # reset agent for training 
        self.reset_episode()

    def reset_episode(self):
        self.it = 0
        self.noise.reset()

    def act(self, state, learn=True):
        with torch.no_grad():
            action = self.actor(self.tensor(state)).cpu().numpy()
        if learn:
            action += self.noise.sample()
        return np.clip(action, self.action_low, self.action_high)

    def step(self):
        pass 

    def soft_update(self):
        """Soft update of target network
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

    def tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)
