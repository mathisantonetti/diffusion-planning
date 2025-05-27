import torch.nn as nn
import torch
import numpy as np
import diffusion_planning.utils as utils

class BaseInvKinematic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = utils.MLP(*args, **kwargs)

    def forward(self, observations, goals):
        return self.model(torch.cat((observations, goals), dim=-1))

    def loss(self, batch):
        her_start_idx = np.random.randint(0, batch[0].shape[1], batch[0].shape[0])
        her_goal_idx = np.random.randint(her_start_idx, batch[0].shape[1], batch[0].shape[0])
        return torch.mean((self(batch[0][:,her_start_idx,:], batch[0][:,her_goal_idx,:]) - batch[1][:,her_start_idx,:])**2)