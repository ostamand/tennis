# pylint: skip-file
from collections import deque
import numpy as np

def train(env, agent, episodes=1000, steps=200, 
          log_each=10, save_thresh=90.0, save_file='saved_models/ddpg.ckpt'):
    scores = deque(maxlen=100)
    last_saved = 0
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
            mean = np.mean(scores)
            summary += f', Score: {mean:.2f}'

            if mean > save_thresh and mean > last_saved:
                summary += ' (saved)'
                last_saved = score 
                agent.save(save_file)

        print(summary)