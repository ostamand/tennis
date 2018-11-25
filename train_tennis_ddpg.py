# pylint: skip-file
from ddpg.agent import Agent
from unity_env import UnityEnv
from tennis.model import Actor, Critic
import numpy as np
from collections import deque

# fill replay buffer with rnd actions
def init_replay_buffer(env, agent, steps):
    state = env.reset()
    for _ in range(steps):
        action =  (np.random.rand(2,2)*2)-1 # btwn -1..1
        next_state, reward, done = env.step(action)
        agent.replay_buffer.add(state.reshape(-1), action.reshape(-1), np.max(reward), next_state.reshape(-1), np.max(done))
        state = next_state
        if done.any():
            state = env.reset()

def train(env, agent, episodes, steps):
    scores = deque(maxlen=100)
    for ep_i in range(episodes):
        agent.reset_episode()
        state = env.reset()
        score = np.zeros(env.num_agents)
        for step_i in range(steps):
            action = agent.act(state.reshape(-1))
            next_state, reward, done = env.step(action.reshape(2,-1))
            score += reward
            # step agent
            # if one is done both are done 
            agent.step(state.reshape(-1), action.reshape(-1), np.max(reward), next_state.reshape(-1), np.max(done))
            state = next_state
            if done.any():
                break
        # log episode results
        scores.append(score.max())
        summary = f'Episode: {ep_i+1}/{episodes}, Steps: {agent.it:d}, Score Agt. #1: {score[0]:.2f}, Score Agt. #2: {score[1]:.2f}'
        if len(scores) >= 100:
            mean = np.mean(scores)
            summary += f', Score: {mean:.3f}'
        print(summary)

if __name__ == '__main__':
    # hyperparameters
    episodes = 2000
    steps = 2000

    # environment 
    env = UnityEnv(no_graphics=False)
    state_size = env.state_size*2
    action_size = env.action_size*2

    # agent
    agent = Agent(
        state_size, action_size, Actor, Critic,
        lrate_critic=1e-3,
        lrate_actor=1e-4,
        tau=0.01,
        buffer_size=1e6,
        batch_size=256,
        gamma=0.99,
        exploration_mu=0.0,
        exploration_theta=0.15,
        exploration_sigma=0.20,
        seed=np.random.randint(1000),
        update_every=1, 
        update_repeat=1,
        weight_decay=0, 
    )

    # start with rnd actions
    init_replay_buffer(env, agent, int(1e4))

    train(env, agent, episodes, steps)
    


