import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Adapted from https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
class ActorCritic(nn.Module):
    def __init__(self, n_state_features, n_actions, n_hidden):
        super(ActorCritic, self).__init__()
        
        # Input state, output state value
        self.critic_network = torch.nn.Sequential(
            nn.Linear(n_state_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

        # Input state, output policy (distribution over actions)
        self.actor_network = torch.nn.Sequential(
            nn.Linear(n_state_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
            nn.Softmax()
        )
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        state_value = self.critic_network(state)
        policy = self.actor_network(state)

        return state_value, policy