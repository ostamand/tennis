# pylint: skip-file
from collections import deque
import numpy as np
from tensorboardX import SummaryWriter

from unity_env import UnityEnv
from ddpg.agent import Agent
from tennis.model import Actor, Critic

if __name__ == "__main__":
    # TODO set seed?
    # TODO share OUNoise?

    # Hyperparameters
    
    episodes = 2000 # total number of episodes to run
    steps = 2000 # maximum number of steps per episode
    upd_every = 1 # update agents every # of steps 
    batch_size = 32

    expl_theta = 0.15 
    expl_sigma = 0.2

    lrate_actor = 10e-4
    lrate_critic = 10e-3

    tau=0.01

    # environment
    env = UnityEnv()

    # create two agents
    agents = []
    for i in range(env.num_agents):
        agents.append(
            Agent(
                env.state_size*2, env.action_size, Actor, Critic, 
                exploration_sigma=expl_sigma,
                exploration_theta=expl_theta,
                lrate_actor=lrate_actor, 
                lrate_critic=lrate_critic,
                update_every=upd_every,
                batch_size=batch_size, 
                tau=tau
            )
        )

    # second agent share same critic network
    # agents[-1].critic = agents[0].critic
    # agents[-1].critic_target = agents[0].critic_target

    # common replay buffer for both agents
    # agents[-1].replay_buffer = agents[0].replay_buffer

    # only one agent will train the critic network 
    # both agent train their actor network
    # agents[-1].train_critic = False 

    # logging
    scores = deque(maxlen=100)
    writer = SummaryWriter()

    it = 0
    action = np.zeros((env.num_agents, env.action_size))
    for ep_i in range(episodes):
        state = env.reset()
        # reset agents episode 
        for agent in agents: 
            agent.reset_episode()
    
        score = np.zeros(env.num_agents)
        for step_i in range(steps):
            # get agents actions from env. state
            for i, agent in enumerate(agents):
                action[i] = agent.act(state)

            # step environment with selected actions
            next_state, reward, done = env.step(action)
            score += reward
            it += 1

            # step agents 
            for i, agent in enumerate(agents):
                agent.step(state, action[i], reward[i], next_state, done[i])

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



    











