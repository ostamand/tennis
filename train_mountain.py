# pylint: skip-file
from ddpg.agent import Agent
from mountain.model import Actor, Critic
from ddpg.train_agent import train
import gym 

if __name__ == '__main__':
    update_every = 4
    lrate_critic=10e-3
    lrate_actor=10e-4
    tau=0.001 # soft update rate
    gamma = 0.99 # discount factor

    # exploration
    exploration_mu = 0.0
    exploration_theta = 0.15
    exploration_sigma = 0.20

    # replay buffer
    buffer_size = 1e5
    batch_size = 64

    # training 
    episodes = 1000
    steps = 500
    log_each = 10

    # environment 
    env = gym.make('MountainCarContinuous-v0')
    state = env.reset()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # TODO rnd seed
    # agent
    agent = Agent(
        state_size, action_size, Actor, Critic, 
        lrate_critic=lrate_critic,
        lrate_actor=lrate_actor,
        tau=tau,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_mu=exploration_mu,
        exploration_theta=exploration_theta,
        exploration_sigma=exploration_sigma
    )

    train(
        env, agent,
        episodes=episodes, steps=steps, log_each=log_each
    )