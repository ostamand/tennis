import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ddpg.noise import OUNoise
from ddpg.replay_buffer import ReplayBuffer

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Agent():
    def __init__(self, state_size, action_size, actor, critic,
                 action_low=-1.0, action_high=1.0,
                 lrate_critic=10e-3, lrate_actor=10e-4, tau=0.001,
                 buffer_size=1e5, batch_size=64, gamma=0.99,
                 exploration_mu=0.0, exploration_theta=0.15,
                 exploration_sigma=0.20,
                 seed=None):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.seed = seed if seed else 0
        self.lrate_critic = lrate_critic
        self.lrate_actor = lrate_actor
        self.tau = tau
        self.gamma = gamma
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

        # optimizer
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lrate_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lrate_critic)

        # noise
        self.noise = OUNoise(action_size, exploration_mu, exploration_theta, exploration_sigma)

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.device, self.buffer_size, self.batch_size)

        # reset agent for training
        self.reset_episode()
        self.it = 0

    def reset_episode(self):
        self.noise.reset()

    def act(self, state, learn=True):
        with torch.no_grad():
            action = self.actor(self.tensor(state)).cpu().numpy()
        if learn:
            action += self.noise.sample()
        return np.clip(action, self.action_low, self.action_high)

    def step(self, state, action, reward, next_state, done):
        #pylint: disable=line-too-long
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.it += 1
        if self.it < self.batch_size:
            return
        # learn from mini-batch of replay buffer 
        state_b, action_b, reward_b, next_state_b, done_b = self.replay_buffer.sample()

        # calculate td target 
        with torch.no_grad():
            y_b = reward_b.unsqueeze(1) + self.gamma * \
             self.critic_target(next_state_b, self.actor_target(next_state_b)) * (1-done_b.unsqueeze(1))

        # update critic
        critic_loss = F.smooth_l1_loss(self.critic(state_b, action_b), y_b)
        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # update actor
        action = self.actor(state_b)
        actor_loss = -self.critic(state_b, action).mean()
        self.actor.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # soft update networks
        self.soft_update()

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
