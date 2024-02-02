import torch
from models.ffn import FeedForwardNN
from envs.navigation import GridWorldEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


obs_dim = 2
act_dim = 4
num_agents = 2

# Define the actor and critic models
actors = [FeedForwardNN(obs_dim, act_dim) for _ in range(num_agents)]

# Load the pre-trained weights for each agent
for agent_idx in range(num_agents):
    actors[agent_idx].load_state_dict(torch.load(f'/home/angelsylvester/Documents/dynamic-rl/marl_mpe/checkpoints/ppo_actor_10_agent_0.pth'))

for actor in actors: 
    actor.eval()

env = GridWorldEnv(render_mode="rgb_array", num_agents=num_agents)

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
        state_tensor = states[agent_idx]['agent']

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
    if all(dones):
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
