from diffusion_planning.models.dit_encoder import DiT1D_Traj_Time_Encoder
from diffusion_planning.planners.comp_diffuser.base import CmpDiffPlanner
import torch


#x, t = torch.ones(13, 11, 5), torch.zeros(13)
#encoder = DiT1D_Traj_Time_Encoder(7, 5, 256)
#print(encoder(x,t).shape)

for i in range(4, 130):
    traj = torch.arange(0, i, 1)[None, :, None] + torch.arange(0, 13000, 1000)[:, None, None]
    #traj = traj.repeat(1, 1, 17)
    print(i)
    planner = CmpDiffPlanner(1, 256, 20.0, 0.1, 3)
    planner.loss(traj)
