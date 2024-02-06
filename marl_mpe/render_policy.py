import torch
import os 
import numpy as np 
from models.ffn import FeedForwardNN
from envs.navigation import GridWorldEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

share_orientation = True 
gifting = False 
# maybe subject to change if model architecture is different 
obs_dim = 10 if share_orientation else 6
act_dim = 5 if gifting else 4 
num_agents = 2

# Define the actor and critic models
actors = [FeedForwardNN(obs_dim, act_dim) for _ in range(num_agents)]

parent_path = "/home/angelsylvester/Documents/dynamic-rl/marl_mpe/checkpoints/simple_pos + gifting_False + time_delayFalse"

# Get a list of all files in the parent path
all_files = os.listdir(parent_path)

# Filter files that match the desired pattern
actor_files = [file for file in all_files if "ppo_actor" in file and file.endswith(".pth")]
print(actor_files)
# Load the pre-trained weights for each agent
for agent_idx in range(num_agents):
    # actors[agent_idx].load_state_dict(torch.load(f'/home/angelsylvester/Documents/dynamic-rl/marl_mpe/checkpoints/ppo_actor_10_agent_0.pth'))
    actors[agent_idx].load_state_dict(torch.load(f'{parent_path}/{actor_files[agent_idx]}'))

for actor in actors: 
    actor.eval()

env = GridWorldEnv(render_mode="rgb_array", num_agents=num_agents, obs_type="simple pos", time_delay = False, nonholonomic=True)

# Initialize the environment
states, _ = env.reset()
done = [False] * num_agents

# Set up the plot for animation
fig, ax = plt.subplots()
im = ax.imshow(env.render().astype(float), animated=True)

# Function to update the plot at each frame
def update(frame):
    actions = []
    for agent_idx in range(num_agents):
        if share_orientation: 
            state_tensor = np.concatenate((states[agent_idx]['agent'],states[agent_idx]['ultrasonic'],states[agent_idx]['neigh_orient']))
        else: 
            state_tensor = np.concatenate((states[agent_idx]['agent'], states[agent_idx]['ultrasonic']))

        # Get action from the actor model
        with torch.no_grad():
            mean = actors[agent_idx](state_tensor)
            dist = torch.distributions.Categorical(logits=mean)
            # Sample an action from the distribution
            action = dist.sample()
            actions.append(action.item())

    # Take a step in the environment
    next_state, rewards, dones, _, infos = env.step(actions)

    # Render the environment
    screen = env.render().astype(float)

    # Update the plot
    im.set_array(screen)


    # Handle episode completion by resetting the environment
    if dones:
        next_states, _ = env.reset()
    else:
        next_states = next_state

    return im,

# Set up the animation with a fixed number of frames
ani = FuncAnimation(fig, update, frames=100, interval=200, blit=True)

# Save the animation as a GIF
ani.save('policy_animation.gif', writer='imagemagick', fps=5)

# Display the animation
plt.show()

# Close the environment
env.close()
