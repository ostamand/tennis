# pylint: skip-file
from collections import deque
import numpy as np
from tensorboardX import SummaryWriter

from unity_env import UnityEnv
from ddpg.m_agent import Agent
from tennis.model import Actor, Critic

if __name__ == "__main__":
    # TODO set seed?
    # TODO share OUNoise?

    # Hyperparameters
    
    episodes = 2000 # total number of episodes to run
    steps = 2000 # maximum number of steps per episode
    upd_every = 1 # update agents every # of steps 
    batch_size = 128

    expl_theta = 0.15 
    expl_sigma = 0.2

    lrate_actor = 1e-3
    lrate_critic = 1e-3

    tau=0.02

    # environment
    env = UnityEnv()

    agent = Agent(
        env.state_size*2, env.action_size, Actor, Critic, 
        exploration_sigma=expl_sigma,
        exploration_theta=expl_theta,
        lrate_actor=lrate_actor, 
        lrate_critic=lrate_critic,
        update_every=upd_every,
        batch_size=batch_size, 
        tau=tau
    )
  
    # logging
    scores = deque(maxlen=100)
    writer = SummaryWriter()

    it = 0
    action = np.zeros((env.num_agents, env.action_size))
    for ep_i in range(episodes):
        state = env.reset()
        # reset agents episode 
        agent.reset_episode()
        score = np.zeros(env.num_agents)
        for step_i in range(steps):
            # get agents actions from env. state
            action[0] = agent.act(state, 0)
            action[1] = agent.act(state, 1)

            # step environment with selected actions
            next_state, reward, done = env.step(action)
            score += reward
            it += 1

            # step first agent 
            agent.step(state, action[0], reward[0], next_state, done[0], 0)
            # step second agent
            agent.step(state, action[1], reward[1], next_state, done[1], 1)

            state = next_state
            # if any of the two agents is done stop 
            if done.any(): 
                break 

        # log episode results
        scores.append(score.max())
        summary = f'Episode: {ep_i+1}/{episodes}, Steps: {it:d}, Score Agt. #1: {score[0]:.2f}, Score Agt. #2: {score[1]:.2f}'
        if len(scores) >= 100:
            mean = np.mean(scores)
            summary += f', Score: {mean:.2f}'
            writer.add_scalar('data/score', mean, ep_i)

        print(summary)



    











