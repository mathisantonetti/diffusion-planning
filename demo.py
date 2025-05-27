"""
This scripts demonstrates how to train Comp Diffuser on the OGBench humanoidmaze medium stitch environment.
"""

import ogbench
import torch
import numpy as np
import imageio
from pathlib import Path
from torch.utils.data import DataLoader

from diffusion_planning.planners import CmpDiffPlanner
from diffusion_planning.invkinematics import BaseInvKinematic
from diffusion_planning.policies import BasePolicy
from diffusion_planning.utils import transforms_dataset_ogbench

# Create a directory to store the video of the evaluation
output_directory = Path("/home/mantonetti/Documents/GCRL/outputs/")
output_directory.mkdir(parents=True, exist_ok=True)

num_epochs = 50

# Make an environment and datasets (they will be automatically downloaded).
dataset_name = 'humanoidmaze-medium-stitch-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)
print("dataset loaded")

# Train your offline goal-conditioned RL agent on the dataset.
train_dataset, observation_dim, action_dim = transforms_dataset_ogbench(train_dataset)
val_dataset, _, _ = transforms_dataset_ogbench(val_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

policy = BasePolicy(planner=CmpDiffPlanner, invkinematic=BaseInvKinematic, planner_args=[[observation_dim, 256, 0.1, 20.0, 4], {}], invkinematic_args=[[2*observation_dim, [512, 1024, 1024, 512, 256], action_dim, False], {}])

optimizer = torch.optim.Adam([
        {"params": policy.planner.parameters()},
        {"params": policy.invkinematic.parameters(), "lr": 1e-4},
    ], lr=2e-4)


for epoch in range(num_epochs):

    # Train one epoch
    train_loss, step = 0.0, 0
    for batch in train_dataloader:
        #print(batch[0].shape, batch[1].shape)
        optimizer.zero_grad()
        loss = policy.loss(batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        step += 1

    # Evaluate
    policy.eval()
    val_loss, val_step = 0.0, 0
    for batch in val_dataloader:
        val_loss += policy.loss(batch).item()
        step += 1

    print("epoch :", epoch, "| train loss :", train_loss/step, "| val loss :", val_loss/step)

"""
# Evaluate the agent.
for task_id in [1, 2, 3, 4, 5]:

    # Reset the environment and set the evaluation task.
    ob, info = env.reset(
        options=dict(
            task_id=task_id,  # Set the evaluation task. Each environment provides five
                              # evaluation goals, and `task_id` must be in [1, 5].
            render_goal=True,  # Set to `True` to get a rendered goal image (optional).
        )
    )

    frames = []
    frames.append(env.render())

    goal = info['goal']  # Get the goal observation to pass to the agent.
    goal_rendered = info['goal_rendered']  # Get the rendered goal image (optional).

    done = False
    while not done:
        action = env.action_space.sample()  # Replace this with your agent's action.
        ob, reward, terminated, truncated, info = env.step(action)  # Gymnasium-style step.
        # If the agent reaches the goal, `terminated` will be `True`. If the episode length
        # exceeds the maximum length without reaching the goal, `truncated` will be `True`.
        # `reward` is 1 if the agent reaches the goal and 0 otherwise.
        done = terminated or truncated
        frames.append(env.render())  # Render the current frame (optional).

    success = info['success']  # Whether the agent reached the goal (0 or 1).
                               # `terminated` also indicates this.
    print(success)
    
    fps = env.metadata["render_fps"]
    # Encode all frames into a mp4 video.
    video_path = output_directory / Path(str(task_id)+ ".mp4")
    imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
"""