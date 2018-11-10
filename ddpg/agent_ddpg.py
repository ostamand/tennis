import torch 

class Agent():
    def __init__(self, env, actor, critic):
        self.env = env

        self.state = env.reset()

    def step(self):
        pass 