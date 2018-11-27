# pylint: skip-file
from ddpg.agent import Agent
from tennis.model import Actor, Critic
from unity_env import UnityEnv
import numpy as np
import argparse

def watch(env, agent, episodes):
    scores = []
    for ep_i in range(episodes):
        score = np.zeros((2,2))
        state = env.reset(train=False)
        while True:
            action = agent.act(state.reshape(-1), learn=False)
            next_state, reward, done = env.step(action.reshape(2, -1))
            score += reward
            state = next_state
            if done.any():
                break
        scores.append(np.max(score))
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', '-a', default='saved_models/tennis_ddpg.ckpt')
    args = parser.parse_args()

    # create environment
    env = UnityEnv(no_graphics=False)
    state_size = env.state_size*2
    action_size = env.action_size*2 

    # restore agent checkpoint
    agent = Agent(state_size, action_size, Actor, Critic, restore=args.agent)

    # watch agent 
    scores = watch(env, agent, 10)

    print(f'Average score over 10 episodes: {np.mean(scores):.2f}')

    env.close()
