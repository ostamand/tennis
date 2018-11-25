import torch
import torch.nn as nn 
import torch.nn.functional as F

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=None):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = torch.device(DEVICE)
        if seed is not None:
            torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256+action_size, 128)
        self.fc3 = nn.Linear(128, 1)
        self.to(self.device)

        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, low=-1.0, high=1.0, seed=None):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.low = low
        self.high = high
        self.device = torch.device(DEVICE)
        if seed is not None:
            torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.to(self.device)

        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.clamp(torch.tanh(self.fc3(x)), self.low, self.high)