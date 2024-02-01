# import torch
# import torch.nn.functional as F
# from models.ffn import FeedForwardNN  # Replace with your actual module

# from envs.navigation import GridWorldEnv
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import time

# start_time = time.time()
# time_out = 5

# obs_dim = 2
# act_dim = 4
# num_agents = 2

# # Define the actor and critic models
# actors = [FeedForwardNN(obs_dim, act_dim) for _ in range(num_agents)]  # Replace with the appropriate sizes
# critics = [FeedForwardNN(obs_dim, 1) for _ in range(num_agents)]  # Replace with the appropriate size

# # Load the pre-trained weights
# # actor_weights = torch.load('/home/angelsylvester/Documents/dynamic-rl/marl_mpe/ppo_actor.pth')
# # critic_weights = torch.load('/home/angelsylvester/Documents/dynamic-rl/marl_mpe/ppo_critic.pth')
# # actor.load_state_dict(actor_weights)
# # critic.load_state_dict(critic_weights)

# # Load the pre-trained weights for each agent
# for agent_idx in range(num_agents):
#     actors[agent_idx].load_state_dict(torch.load(f'/home/angelsylvester/Documents/dynamic-rl/marl_mpe/ppo_actor.pth'))
#     critics[agent_idx].load_state_dict(torch.load(f'/home/angelsylvester/Documents/dynamic-rl/marl_mpe/ppo_critic.pth'))


# for actor in actors: 
#     actor.eval()

# env = GridWorldEnv(render_mode = "rgb_array", num_agents = num_agents)

# # Initialize the environment
# states, _ = env.reset()
# done = [False] * num_agents

# while not all(done) and time.time() - start_time < time_out:
#     actions = []
#     for agent_idx in range(num_agents):
#         state_tensor = states[agent_idx]['agent']

#         # Get action from the actor model
#         with torch.no_grad():
#             mean = actors[agent_idx](state_tensor)
#             dist = torch.distributions.Categorical(logits=mean)
#             # Sample an action from the distribution
#             action = dist.sample()
#             actions.append(action.item())

#     # Take a step in the environment
#     next_state, rewards, dones, _, infos = env.step(actions)  # Assuming step method takes agent_idx

#     # Render the environment (if needed)
#     # env.render()
#     screen = env.render()

#     # Convert the screen array to a numeric type
#     screen = screen.astype(float)

#     plt.imshow(screen)
#     print('showing')


#     for i in range(num_agents): 
#         states[agent_idx]['agent'] = next_state[agent_idx]['agent']

#     # Display the plot
#     plt.show()

# # Close the environment
# env.close()

import torch
from models.ffn import FeedForwardNN
from envs.navigation import GridWorldEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

start_time = time.time()
time_out = 5

obs_dim = 2
act_dim = 4
num_agents = 2

# Define the actor and critic models
actors = [FeedForwardNN(obs_dim, act_dim) for _ in range(num_agents)]

# Load the pre-trained weights for each agent
for agent_idx in range(num_agents):
    actors[agent_idx].load_state_dict(torch.load(f'/home/angelsylvester/Documents/dynamic-rl/marl_mpe/ppo_actor.pth'))

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

    if time.time() - start_time > 10: 
        print('time out' )
        return 

    return im,

# Set up the animation
ani = FuncAnimation(fig, update, frames=None, interval=200, blit=True)

# Save the animation as a GIF
ani.save('policy_animation.gif', writer='imagemagick', fps=5)

# Display the animation
plt.show()

# Close the environment
env.close()
