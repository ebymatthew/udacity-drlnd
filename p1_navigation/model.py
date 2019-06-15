import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, h1_units=64, h2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(state_size,h1_units)
        self.fc2 = nn.Linear(h1_units,h2_units)
        self.fc3 = nn.Linear(h2_units,action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        activation1 = F.relu(self.fc1(state))
        activation2 = F.relu(self.fc2(activation1))
        y = self.fc3(activation2)
        return y
