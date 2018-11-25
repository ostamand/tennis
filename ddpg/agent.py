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
                 buffer_size=1e5, batch_size=64, gamma=0.99,
                 exploration_mu=0.0, exploration_theta=0.15, noise_decay=1., 
                 exploration_sigma=0.20, restore=None, weight_decay=0.,
                 update_every=1, update_repeat=1, seed=None):
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
        self.weight_decay = weight_decay
        self.update_repeat = update_repeat
        self.noise_decay = noise_decay

        # actors networks
        self.actor = actor(state_size, action_size,
                           low=action_low, high=action_high, seed=self.seed)
        self.actor_target = actor(state_size, action_size,
                                  low=action_low, high=action_high, seed=self.seed)

        # critic networks
        self.critic = critic(state_size, action_size, seed=self.seed)
        self.critic_target = critic(state_size, action_size, seed=self.seed)

        # restore networks if needed
        if restore is not None:
            checkpoint = torch.load(restore, map_location=DEVICE)
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor_target.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic'])

        # optimizer
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lrate_actor, weight_decay=self.weight_decay)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lrate_critic, weight_decay=self.weight_decay)

        # noise
        self.noise = OUNoise(action_size, exploration_mu, exploration_theta, exploration_sigma)
        self.noise_scale = 1.0

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.device, self.buffer_size, self.batch_size)

        # reset agent for training
        self.reset_episode()
        self.it = 0

    def reset_episode(self):
        self.noise.reset()

    def act(self, state, learn=True):
        if not learn:
            self.actor.eval()

        with torch.no_grad():
            action = self.actor(self.tensor(state)).cpu().numpy()
        if learn:
            action += self.noise.sample() * self.noise_scale
            self.noise_scale = max(self.noise_scale * self.noise_decay, 0.01)

        self.actor.train()
        return np.clip(action, self.action_low, self.action_high)

    def save(self, path):
        dirn = os.path.dirname(path)
        if not os.path.exists(dirn):
            os.mkdir(dirn)
        params = {}
        params['actor'] = self.actor.state_dict()
        params['critic'] = self.critic.state_dict()
        torch.save(params, path)

    def step(self, state, action, reward, next_state, done):
        #pylint: disable=line-too-long
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.it += 1
        if self.it < self.batch_size or self.it % self.update_every != 0:
            return
        for _ in range(self.update_repeat):
            self.learn()

    def learn(self):
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
        # critic only if trained
        # actor always
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
