# pylint: skip-file
from ddpg.agent import Agent
from tennis.model import Actor, Critic
from unity_env import UnityEnv
import numpy as np

def watch(env, agent):
    state = env.reset(train=False)
    score = np.zeros(2,2)
    while True:
        action = agent.act(state.reshape(-1), learn=False)
        next_state, reward, done = env.step(action.reshape(2, -1))
        score += reward
        state = next_state
        if done.any():
            break
    return np.max(score)

if __name__ == '__main__':
    # create environment
    env = UnityEnv(no_graphics=False)
    state_size = env.state_size*2
    action_size = env.action_shape*2 

    # restore agent checkpoint
    agent = Agent(state_size, action_size, Actor, Critic, restore='saved_models/tennis_ddpg.ckpt')

    # watch agent 
    score = watch(env, agent)

    print(f'Episode score: {score:.2f}')
