import torch
import os 
import numpy as np 
from models.ffn import FeedForwardNN
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from harvest import HarvestEnv

obs_dim = 687 
act_dim = 8 
num_agents = 3

# Define the actor and critic models
actors = [FeedForwardNN(obs_dim, act_dim) for _ in range(num_agents)]

parent_path = "/home/angelsylvester/Documents/dynamic-rl/marl_mpe/checkpoints"

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

env = HarvestEnv(num_agents=3,
                return_agent_actions=True,
                use_collective_reward=True,
                inequity_averse_reward=True,
                alpha=0.0,
                beta=0.0)


# Initialize the environment
states = env.reset()
done = [False] * num_agents

# Set up the plot for animation
fig, ax = plt.subplots()

print(f'env rendered: {env.render()}')
im = ax.imshow(env.render().astype(float), animated=True)

# Function to update the plot at each frame
def update(frame):
    actions = []
    for agent_idx in range(num_agents):
        obs = states 

        agent_observation = {
        "curr_obs": obs[agent_idx]["curr_obs"].flatten(),
        "other_agent_actions": obs[agent_idx]["other_agent_actions"],
        "visible_agents": obs[agent_idx]["visible_agents"],
        "prev_visible_agents": obs[agent_idx]["prev_visible_agents"]
        }

        # Concatenate the observation components
        concatenated_observation = np.concatenate((
            agent_observation["curr_obs"],
            agent_observation["other_agent_actions"],
            agent_observation["visible_agents"],
            agent_observation["prev_visible_agents"]
        ))


        state_tensor = concatenated_observation

        # Get action from the actor model
        with torch.no_grad():
            mean = actors[agent_idx](state_tensor)
            dist = torch.distributions.Categorical(logits=mean)
            # Sample an action from the distribution
            action = dist.sample()
            actions.append(action.item())

    # Take a step in the environment
    next_state, rewards, dones, infos = env.step(actions)

    # Render the environment
    screen = env.render().astype(float)

    # Update the plot
    im.set_array(screen)


    # Handle episode completion by resetting the environment
    if dones:
        next_states = env.reset()
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
