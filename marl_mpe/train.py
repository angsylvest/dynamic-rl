"""
    This file is the executable for running PPO. It is based on this medium article: 
    https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gym
import sys
import torch

from arguments import get_args
from models.ppo import PPO
from models.ffn import FeedForwardNN
from eval_policy import eval_policy

from envs.navigation import GridWorldEnv

def train(env, hyperparameters, actor_model, critic_model, num_agents, policy_type):
    """
        Trains the model.

        Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training

        Return:
            None
    """	
    
    print(f"Training", flush=True)

    # Create a model for PPO (converted to IPPO)
    model = PPO(policy_class=FeedForwardNN, env=env, num_agents = num_agents, policy_type=policy_type, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    for i in range(num_agents): 
        if actor_model != '' and critic_model != '':
            print(f"Loading in {actor_model} and {critic_model}...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.critic.load_state_dict(torch.load(critic_model))
            print(f"Successfully loaded.", flush=True)
        elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
            print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
            sys.exit(0)
        else:
            print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=200_000_000)

def test(env, actor_model, num_agents):
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardNN(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True)

def main(args, mode, num_agents, obs_type, time_delay, policy_type, nonholonomic):
    """
        The main function to run.

        Parameters:
            args - the arguments parsed from command line

        Return:
            None
    """
    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
                'timesteps_per_batch': 2048, 
                'max_timesteps_per_episode': 200, 
                'gamma': 0.99, 
                'n_updates_per_iteration': 10,
                'lr': 3e-4, 
                'clip': 0.2,
                'render': True,
                'render_every_i': 10
            }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym and have both continuous
    # observation and action spaces.
    env = GridWorldEnv(num_agents=num_agents, obs_type = obs_type, time_delay=time_delay, nonholonomic=nonholonomic) # gym.make('Pendulum-v0')

    # Train or test, depending on the mode specified
    if  mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model, num_agents = num_agents, policy_type = policy_type)
    else:
        test(env=env, actor_model=args.actor_model, num_agents = num_agents)

if __name__ == '__main__':
    mode = "train"
    num_agents = 2
    obs_types = ["simple pos", "simple pos and local occupancy", "simple pos and vector occupancy"]
    obs_type = obs_types[0]
    time_delay = False
    nonholonomic = True
    policy_type = f"simple_pos + time_delay_{time_delay}" # just way to label policies 
    args = get_args() # Parse arguments from command line
    main(args, mode, num_agents, obs_type, time_delay, policy_type, nonholonomic)