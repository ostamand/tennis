import os
import unittest
import pdb

import gym
import torch

from ddpg.agent import Agent 
from mountain.model import Actor, Critic

def fill_replay_buffer(env, agent, n=100):
    state = env.reset()
    for i in range(n):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        if done:
            state = env.reset()
        else:
            state = next_state
    agent.it = i+1

class TestDDPGAgent(unittest.TestCase):

    def setUp(self):
        self.env = gym.make('MountainCarContinuous-v0')
        self.state = self.env.reset()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agent_params = (self.state_size, self.action_size, Actor, Critic)

    def test_can_create_agent(self):
        agent = Agent(*self.agent_params)
        self.assertAlmostEqual(agent.actor.fc1.weight[0, 0].item(),
                               agent.actor_target.fc1.weight[0, 0].item())
        self.assertAlmostEqual(agent.critic.fc1.weight[0, 0].item(),
                               agent.critic_target.fc1.weight[0, 0].item())

    def test_can_agent_act(self):
        agent = Agent(*self.agent_params)
        action = agent.act(self.state)
        self.assertIsNotNone(action)
        next_state, _, _, _ = self.env.step(action)
        self.assertIsNotNone(next_state)

    def test_can_add_stuff_to_replay_buffer(self):
        agent = Agent(*self.agent_params)
        action = agent.act(self.state)
        next_state, reward, done, _ = self.env.step(action)
        agent.replay_buffer.add(self.state, action, reward, next_state, done)
        self.assertGreater(len(agent.replay_buffer), 0)

    def test_can_get_batch_from_replay_buffer(self):
        agent = Agent(*self.agent_params)
        fill_replay_buffer(self.env, agent, n=100)
        self.assertEqual(len(agent.replay_buffer), 100)
        state_b, action_b, reward_b, next_state_b, done_b = agent.replay_buffer.sample()
        self.assertEqual(state_b.shape[0], agent.batch_size)
        self.assertEqual(action_b.shape[0], agent.batch_size)
        self.assertEqual(reward_b.shape[0], agent.batch_size)
        self.assertEqual(next_state_b.shape[0], agent.batch_size)
        self.assertEqual(done_b.shape[0], agent.batch_size)

    def test_can_step_agent(self):
        agent = Agent(*self.agent_params)
        fill_replay_buffer(self.env, agent, n=100)
        state = self.env.reset()
        action = agent.act(state)
        next_state, reward, done, _ = self.env.step(action)
        agent.step(state, action, reward, next_state, done)

    def test_can_save_restore_agent(self):
        save_f = 'saved_models/test.ckpt'
        agent = Agent(*self.agent_params)
        agent.save(save_f)
        self.assertTrue(os.path.exists(save_f))
        agent2 = Agent(*self.agent_params, restore=save_f)
        self.assertEqual(agent.actor.fc1.weight[0, 0], agent2.actor.fc1.weight[0, 0])
        self.assertEqual(agent.actor_target.fc1.weight[0, 0], agent2.actor_target.fc1.weight[0, 0])
        self.assertEqual(agent.critic.fc1.weight[0, 0], agent2.critic.fc1.weight[0, 0])
        self.assertEqual(agent.critic_target.fc1.weight[0, 0], agent2.critic_target.fc1.weight[0, 0])
        