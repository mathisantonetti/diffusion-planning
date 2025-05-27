import torch.nn as nn

class BasePlanner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, init_states, goal_states):
        pass

    def loss(self, batch):
        pass