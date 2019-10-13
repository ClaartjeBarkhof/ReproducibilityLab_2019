import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):

    def __init__(self, n_state_features, n_actions, n_hidden, device):
        super(Actor, self).__init__()
        self.device = device

        self.n_h1 = n_hidden[0]
        self.n_h2 = n_hidden[1]

        self.actor_network = torch.nn.Sequential(
            nn.Linear(n_state_features, self.n_h1),
            nn.ReLU(),
            nn.Linear(self.n_h1, self.n_h2),
            nn.ReLU(),
            nn.Linear(self.n_h2, n_actions),
            nn.Softmax()
        )

    def forward(self, state):
        state = torch.FloatTensor(state).to(self.device)
        policy = self.actor_network(state)
        return Categorical(policy)


class ValueFunction(nn.Module):
    def __init__(self, n_state_features, n_hidden, device):
        super(ValueFunction, self).__init__()
        self.device = device

        self.n_h1 = n_hidden[0]
        self.n_h2 = n_hidden[1]

        self.critic_network = torch.nn.Sequential(
            nn.Linear(n_state_features, self.n_h1),
            nn.ReLU(),
            nn.Linear(self.n_h1, self.n_h2),
            nn.ReLU(),
            nn.Linear(self.n_h2, 1)
        )

    def forward(self, state):
        state = torch.FloatTensor(state).to(self.device)
        state_value = self.critic_network(state)
        return state_value


class QValueFunction(nn.Module):
    def __init__(self, n_state_features, n_actions, device):
        super(ValueFunction, self).__init__()
        self.device = device

        self.critic_network = torch.nn.Sequential(
            nn.Linear(n_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, state):
        state = torch.FloatTensor(state).to(self.device)
        state_value = self.critic_network(state)
        return state_value


