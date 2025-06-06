import torch.nn as nn
import numpy as np

# TODO: reorganize the full architecture to separate the roles of policy, planner and inverse kinematics.

class BasePolicy(nn.Module):
    def __init__(self, planner=None, planner_args=[], invkinematic=None, invkinematic_args=[]):
        super().__init__()
        if planner is None:
            self.planner = None
        else:
            self.planner = planner(*planner_args[0], **planner_args[1])
        
        if invkinematic is None:
            self.invkinematic = None
        else:
            self.invkinematic = invkinematic(*invkinematic_args[0], **invkinematic_args[1])

        self.plan = []

    def loss(self, batch):
        if self.planner is None:
            if self.invkinematic is None:
                raise NotImplementedError
            return self.invkinematic.loss(batch)
        elif self.invkinematic is None:
            return self.planner.loss(batch)
        else:
            return self.planner.loss(batch) + self.invkinematic.loss(batch)
        
    def update(self, info):
        self.plan = self.planner(info["init"], info["goal"])
        #print(self.plan)

    def select_action(self, info):
        if self.planner is None:
            raise NotImplementedError
        else:
            if len(self.plan) == 0:
                self.update(info)
            if self.invkinematic is None:
                return self.plan.pop(0).detach().cpu().numpy()[0]
            else:
                #print(self.invkinematic(info["init"], self.plan.pop(0)).detach().cpu().numpy())
                return self.invkinematic(info["init"], self.plan.pop(0)).detach().cpu().numpy()[0]