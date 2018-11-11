# pylint: skip-file
from collections import deque
import numpy as np

def train(env, agent, episodes=1000, steps=200, log_each=10):
    scores = deque(maxlen=100)

    for ep_i in range(episodes):
        state = env.reset()
        agent.reset_episode()
        score = 0
        for step_i in range(steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done: 
                break
        scores.append(score)
        summary = f'Episode: {ep_i+1}/{episodes}, Reward: {score:.2f}'
        if len(scores) >= 100:
            summary += f', Score: {np.mean(scores):.2f}'
        print(summary)