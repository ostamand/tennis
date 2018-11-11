import argparse
import gym
from mountain.model import Actor, Critic
from ddpg.agent import Agent

def watch(env, agent, max_steps):
    state = env.reset()
    score = 0 
    for step_i in range(max_steps):
        env.render()
        action = agent.act(state, learn=False)
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break
    env.close()
    print(f'Episode Score: {score:.2f}, Steps: {step_i+1}')

#pylint: disable=invalid-name
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', '-a', default='saved_models/ddpg.ckpt')
    args = parser.parse_args()

    env = gym.make('MountainCarContinuous-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = Agent(state_size, action_size,
                  Actor, Critic, restore=args.agent)

    watch(env, agent, 2000)
