from harvest import HarvestEnv
from models.ppo import PPO
from models.ffn import FeedForwardNN

import globals as globals 
import os 

model_name = ""
num_agents =  5

# just testing 
env = HarvestEnv(num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=False,
                inequity_averse_reward=True,
                alpha=0.0,
                beta=0.0, 
                split_roles=False)

obs_space = env.observation_space
act_space = env.action_space

print(f'obs space {type(obs_space)} and action space {type(act_space)}')

env_type = "social-dilemma" # change to something else if want to use simpler env
mode = "train"
# num_agents = 3
obs_types = ["simple pos", "simple pos and local occupancy", "simple pos and vector occupancy"]
obs_type = obs_types[0]
time_delay = False
nonholonomic = False
gifting = False 
share_orientation = False
# policy_type = f"simple_pos + time_delay_{time_delay}" # just way to label policies 
policy_type = f"env_type_{env_type}_num_agents_{num_agents}_bayes_{globals.bayes}"

checkpoint_dir = f"/home/angelsylvester/Documents/dynamic-rl/marl_mpe/checkpoints/{policy_type}" 

# Create the directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

model = PPO(policy_class=FeedForwardNN, env=env, num_agents = num_agents, policy_type=policy_type, roles = False, checkpoint_dir = checkpoint_dir, gifting = gifting, time_delay = time_delay, share_orientation= share_orientation, env_type = env_type)
model.learn(total_timesteps=200_000_000)