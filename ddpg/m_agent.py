import os
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
                 buffer_size=1e6, batch_size=64, gamma=0.99,
                 exploration_mu=0.0, exploration_theta=0.15,
                 exploration_sigma=0.20, restore=None,
                 update_every=1, seed=None):
        self.num_agents = 2
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.seed = seed if seed else np.random.randint(100)
        self.lrate_critic = lrate_critic
        self.lrate_actor = lrate_actor
        self.tau = tau
        self.gamma = gamma
        self.restore = restore
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.update_every = update_every
        self.device = torch.device(DEVICE)

        # different actor networks
        self.actors = []
        self.actors_opt = []
        self.actors_target = []
        for i in range(self.num_agents):
            # actor network
            seed = np.random.randint(100)
            self.actors.append(
                actor(state_size, action_size,
                      low=action_low, high=action_high, seed=seed)
                )
            # target actor network
            self.actors_target.append(
                actor(state_size, action_size,
                      low=action_low, high=action_high, seed=seed)
                )
            # optimizer
            self.actors_opt.append(optim.Adam(self.actors[i].parameters(), lr=lrate_actor))

        # shared critic network
        seed = np.random.randint(100)
        self.critic = critic(state_size, action_size, seed)
        self.critic_target = critic(state_size, action_size, seed)

        # optimizer
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

    def act(self, state, actor_i, learn=True):
        state = self._get_agent_state(state, actor_i)

        if not learn:
            self.actors[actor_i].eval()

        with torch.no_grad():
            action = self.actors[actor_i](self.tensor(state)).cpu().numpy()
        if learn:
            action += self.noise.sample()

        self.actors[actor_i].train()
        return np.clip(action, self.action_low, self.action_high)

    def step(self, state, action, reward, next_state, done, actor_i):
        #pylint: disable=line-too-long
        state = self._get_agent_state(state, actor_i)
        next_state = self._get_agent_state(next_state, actor_i)

        self.replay_buffer.add(state, action, reward, next_state, done)
        self.it += 1
        if self.it < self.batch_size or self.it % self.update_every != 0:
            return

        # learn from mini-batch of replay buffer
        state_b, action_b, reward_b, next_state_b, done_b = self.replay_buffer.sample()

        # calculate td target
        with torch.no_grad():
            y_b = reward_b.unsqueeze(1) + self.gamma * \
             self.critic_target(next_state_b, self.actors_target[actor_i](next_state_b)) * (1-done_b.unsqueeze(1))

        # update critic
        critic_loss = F.smooth_l1_loss(self.critic(state_b, action_b), y_b)
        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # update actor
        action = self.actors[actor_i](state_b)
        actor_loss = -self.critic(state_b, action).mean()
        self.actors[actor_i].zero_grad()
        actor_loss.backward()
        self.actors_opt[actor_i].step()

        # soft update networks
        self.soft_update(actor_i)

    def soft_update(self, actor_i):
        """Soft update of target network
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, param in zip(self.actors_target[actor_i].parameters(), self.actors[actor_i].parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

    def tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)

    def _get_agent_state(self, state, agent_i):
        if agent_i == 0:
            state = np.concatenate((state[0], state[1]))
        else:
            state = np.concatenate((state[1], state[0]))
        return state
