from harvest import HarvestEnv
from cleanup import CleanupEnv
from models.ppo import PPO
from models.ffn import FeedForwardNN

import globals as globals 
import os 

from utility_funcs import * 

model_name = ""
num_agents =  5

env_used = globals.environment_used
collective_reward = globals.collective_reward
return_agent_actions = globals.return_agent_actions

if env_used == "harvest": 
    # just testing 
    env = HarvestEnv(num_agents=num_agents,
                return_agent_actions=return_agent_actions,
                use_collective_reward=collective_reward,
                inequity_averse_reward=True,
                alpha=0.0,
                beta=0.0, 
                split_roles=False)
    
elif env_used == "clean-up": 
    # just testing 
    
    env = CleanupEnv(num_agents=num_agents,
                return_agent_actions=return_agent_actions,
                use_collective_reward=collective_reward,
                inequity_averse_reward=True,
                alpha=0.0,
                beta=0.0)
    
else: 
    print('ERROR: invalid/implemented env entered')

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
share_orientation = False
# policy_type = f"simple_pos + time_delay_{time_delay}" # just way to label policies 
policy_type = f"env_type_{env_type}_num_agents_{num_agents}_bayes_{globals.bayes}_gifting_{globals.gifting}_envused_{env_used}_collect_{collective_reward}_return_agent_{return_agent_actions}"

if not globals.hpc:
    checkpoint_dir = f"/home/angelsylvester/Documents/dynamic-rl/marl_mpe/checkpoints/{policy_type}" 
else: 
    checkpoint_dir = f"~/dynamic-rl/dynamic-rl/marl_mpe/checkpoints/{policy_type}"

# Create the directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# checkpoint example in hpc: ~/dynamic-rl/msi-scripts/~/dynamic-rl/dynamic-rl/marl_mpe/checkpoints/env_type_social-dilemma_num_agents_5_bayes_True_gifting_False_envused_clean-up/iteration_30
checkpoint_path = "marl_mpe/checkpoints/env_type_social-dilemma_num_agents_5_bayes_True_gifting_False_envused_clean-up/iteration_10" 
actors, critics = extract_paths(checkpoint_path)

# checkpoints = []
# critics = []
model = PPO(policy_class=FeedForwardNN, env=env, num_agents = num_agents, policy_type=policy_type, roles = False, checkpoint_dir = checkpoint_dir, gifting = globals.gifting, time_delay = time_delay, share_orientation= share_orientation, env_type = env_type, checkpoints = actors, critics=critics)
model.learn(total_timesteps=200_000_000)