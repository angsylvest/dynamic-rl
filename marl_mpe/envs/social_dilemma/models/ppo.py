"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gym
import time
import os 
import csv

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

import globals as globals

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, num_agents, policy_type, checkpoint_dir, gifting, time_delay, share_orientation, env_type, roles, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# print(f'current env: {env}')

		# Extract environment information
		self.env = env
		self.env_type = env_type
		self.roles = roles

		if self.env_type == 'social-dilemma': 
			assert(type(env.observation_space) == gym.spaces.dict.Dict)
			# assert(type(env.action_space) == gym.spaces.discrete.Discrete)

		else: 
			print(f'ppo env info: {type(env.observation_space[0])} and act space: {type(env.action_space[0])}')
			assert(type(env.observation_space[0]) == gym.spaces.dict.Dict)
			assert(type(env.action_space[0]) == gym.spaces.discrete.Discrete)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)
		self.gifting = gifting
		self.share_orientation = share_orientation # tune-able
		self.time_delay = time_delay

		# Extract environment information
		self.env = env
		self.env_type = env_type
		# self.obs_dim = env.observation_space[0].shape[0]

		self.bayes = globals.bayes 


		if self.env_type == 'social-dilemma': 
			if self.bayes: 
				# update obs_dim and act_dim 
				obs_size = (
					np.prod(env.observation_space["curr_obs"].shape) +
					np.prod(env.observation_space["other_agent_actions"].shape) +
					np.prod(env.observation_space["visible_agents"].shape) +
					np.prod(env.observation_space["prev_visible_agents"].shape) + 
					np.prod(env.observation_space["bayes_counter"].shape)
				)

			else: 
				# update obs_dim and act_dim 
				obs_size = (
					np.prod(env.observation_space["curr_obs"].shape) +
					np.prod(env.observation_space["other_agent_actions"].shape) +
					np.prod(env.observation_space["visible_agents"].shape) +
					np.prod(env.observation_space["prev_visible_agents"].shape)
				)
			
			# Account for multiple agents by multiplying by the number of agents
			self.obs_dim = obs_size # obs_size_per_agent
			self.act_dim = env.action_space.n
			self.act_dim_follower = env.action_space_roles.n

			# print(f'updated obs_dim: {self.obs_dim}')
		
		else: 
			if self.share_orientation: 
				self.obs_dim = env.observation_space[0]["agent"].shape[0] + env.observation_space[0]["ultrasonic"].shape[0] + env.observation_space[0]["neigh_orient"].shape[0] + env.observation_space[0]["relative_goal_pos"].shape[0]
			else: 
				self.obs_dim = env.observation_space[0]["agent"].shape[0] + env.observation_space[0]["ultrasonic"].shape[0] + env.observation_space[0]["relative_goal_pos"].shape[0]

			if self.time_delay: 
				self.obs_dim += env.observation_space[0]["remaining_steps"].shape[0] + env.observation_space[0]["neigh_remaining_steps"].shape[0]
			
			# self.act_dim = env.action_space[0].shape[0]
			self.act_dim = env.action_space[0].n # is 4 now


		self.policy_type = policy_type
		self.checkpoint_dir = checkpoint_dir

		self.csv_filename_per_agent = f'{self.checkpoint_dir}/csv_per_agent.csv'
		self.csv_filename_cum = f'{self.checkpoint_dir}/csv_cum.csv'

		# Initialize CSV file with column names
		with open(self.csv_filename_per_agent, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['Iteration', 'Agent_ID', 'Average_Reward'])

		# Initialize CSV file with column names
		with open(self.csv_filename_cum, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['Iteration', 'Average_Reward'])


		# separate each by agent (IPPO, will allow for async updates)
		self.num_agents = num_agents 

		if self.roles: 
			self.actors = [policy_class(self.obs_dim, self.act_dim + 2 if i % 2 == 1 else self.act_dim) for i in range(self.num_agents)]
			self.critics = [policy_class(self.obs_dim, 1) for i in range(self.num_agents)]

			self.actor_optims = [Adam(self.actors[i].parameters(), lr=self.lr) for i in range(self.num_agents)]
			self.critic_optims = [Adam(self.critics[i].parameters(), lr=self.lr) for i in range(self.num_agents)]

		else: 
			self.actors = [policy_class(self.obs_dim, self.act_dim) for i in range(self.num_agents)]
			self.critics = [policy_class(self.obs_dim, 1) for i in range(self.num_agents)]

			self.actor_optims = [Adam(self.actors[i].parameters(), lr=self.lr) for i in range(self.num_agents)]
			self.critic_optims = [Adam(self.critics[i].parameters(), lr=self.lr) for i in range(self.num_agents)]

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)


		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")

		
		# Initialize the Matplotlib plot
		plt.ion()  # Turn on interactive mode
		# Create a figure with two subplots
		fig, (ax_total, ax_agents) = plt.subplots(2, 1)
		line_total, = ax_total.plot([], [], label='Total Average Episodic Return')
		ax_total.set_xlabel('Iterations')
		ax_total.set_ylabel('Total Average Episodic Return')
		ax_total.legend()

		agent_lines = {}
		for agent_id in range(self.num_agents):
			line, = ax_agents.plot([], [], label=f'Agent {agent_id} Average Episodic Return')
			agent_lines[agent_id] = line

		ax_agents.set_xlabel('Iterations')
		ax_agents.set_ylabel('Agent Average Reward')
		ax_agents.legend()
		
		
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far

		try: 
			while t_so_far < total_timesteps:                                                                       # ALG STEP 2
				# Autobots, roll out (just kidding, we're collecting our batch simulations here)
				# convert batch to be a dictionary for each agent 
				batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

				print('completed rollout')
				# Calculate how many timesteps we collected this batch
				t_so_far += np.sum(batch_lens)

				# Increment the number of iterations
				i_so_far += 1

				# Logging timesteps so far and iterations so far
				self.logger['t_so_far'] = t_so_far
				self.logger['i_so_far'] = i_so_far


				# update process (will occur for each agent's model)

				for i in range(self.num_agents): 
					# print(f'lens of rel batch info \n batch_obs: {batch_obs[i].shape} \n batch_acts: {batch_acts[i].shape} \n batch_rtgs {batch_rtgs[i].shape} \n  for agent {i}')
					# print(f'appearance of batch_obs: {batch_obs[i]}')
					# Calculate advantage at k-th iteration
					V, _ = self.evaluate(batch_obs[i], batch_acts[i], batch_rtgs[i], i)
					A_k = batch_rtgs[i] - V.detach()                                                                       # ALG STEP 5

					# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
					# isn't theoretically necessary, but in practice it decreases the variance of 
					# our advantages and makes convergence much more stable and faster. I added this because
					# solving some environments was too unstable without it.
					A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

					# This is the loop where we update our network for some n epochs
					for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
						# Calculate V_phi and pi_theta(a_t | s_t)
						V, curr_log_probs = self.evaluate(batch_obs[i], batch_acts[i], batch_rtgs[i],i)

						# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
						# NOTE: we just subtract the logs, which is the same as
						# dividing the values and then canceling the log with e^log.
						# For why we use log probabilities instead of actual probabilities,
						# here's a great explanation: 
						# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
						# TL;DR makes gradient descent easier behind the scenes.
						ratios = torch.exp(curr_log_probs - batch_log_probs[i])

						# Calculate surrogate losses.
						surr1 = ratios * A_k
						surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

						# Calculate actor and critic losses.
						# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
						# the performance function, but Adam minimizes the loss. So minimizing the negative
						# performance function maximizes it.
						actor_loss = (-torch.min(surr1, surr2)).mean()
						critic_loss = nn.MSELoss()(V, batch_rtgs[i])

						# Calculate gradients and perform backward propagation for actor network
						self.actor_optims[i].zero_grad()
						# self.actor_optim.zero_grad()
						actor_loss.backward(retain_graph=True)
						# self.actor_optim.step()
						self.actor_optims[i].step()

						# Calculate gradients and perform backward propagation for critic network
						# self.critic_optim.zero_grad()
						self.critic_optims[i].zero_grad()
						critic_loss.backward()
						# self.critic_optim.step()
						self.critic_optims[i].step()

					# Log actor loss
					self.logger[f'actor_losses'].append(actor_loss.detach())

				# Update the live plot for the total average episodic return
				plot_metrics = {}
				# print(f'batch rews collected info: {self.logger["batch_rews"]}')

				for a in range(self.num_agents):
					plot_metrics[a] = 0
					batch_rews_a = self.logger['batch_rews'][a]
					
					# total_rewards_a = sum(sum(ep_rewards) for ep_rewards in batch_rews_a)
					# total_episodes_a = sum(len(ep_rewards) for ep_rewards in batch_rews_a)
					num_ep = 0 
					total_avgs = 0 
					for ep in batch_rews_a: 
						avg = sum(ep) / len(ep)
						num_ep += 1
						total_avgs += avg 
						# print(f'running avg: {avg} for {sum(ep)} for ep size {len(ep)}')
					# total_rewards_a = (sum(ep_rewards) for ep_rewards in batch_rews_a)
					# total_episodes_a = len(batch_rews_a)
					# total
						

					# print(f'total_rewards_a: {total_rewards_a} and total_episodes_a: {total_rewards_a}')
					
					# plot_metrics[a] = total_rewards_a / total_episodes_a
					plot_metrics[a] = total_avgs / num_ep

				# print(f'plot metrics after calc: {plot_metrics}')

				# Update the live plot for the total average episodic return
				avg_ep_rews_total = np.mean([plot_metrics[a] for a in range(self.num_agents)])
				self._update_live_plot(ax_total, line_total, i_so_far, avg_ep_rews_total, i_so_far, "total", self.csv_filename_cum)

				# Update the live plot for each agent's average episodic return
				for agent_id, line in agent_lines.items():
					avg_ep_rews_agent = plot_metrics[agent_id]
					self._update_live_plot(ax_agents, line, i_so_far, avg_ep_rews_agent, i_so_far, agent_id, self.csv_filename_per_agent)
					# print(f"avg_ep_rews_agent for agent {agent_id}:", avg_ep_rews_agent)


				plt.pause(0.001)

				# Print a summary of our training so far
				self._log_summary()

				# Save our model if it's time
				if i_so_far % self.save_freq == 0:
					print('saving current checkpoint')
					# torch.save(self.actor.state_dict(), './ppo_actor.pth')
					# torch.save(self.critic.state_dict(), './ppo_critic.pth')
					for a in range(self.num_agents): 
						torch.save(self.actors[i].state_dict(), f'{self.checkpoint_dir}/ppo_actor_{i_so_far}_agent_{a}.pth')
						torch.save(self.critics[i].state_dict(), f'{self.checkpoint_dir}/ppo_critic_{i_so_far}_agent_{a}.pth')

		except KeyboardInterrupt:
			print("\nTraining interrupted. Saving current state...")

			# Save dynamic graph or relevant information
			plt.savefig(f'dynamic_graph_interrupted_{self.policy_type}.png')  # Adjust the filename and format as needed
			print("Dynamic graph saved.")

			# Optionally save the model state here if needed


		# Close the plot after training
		plt.ioff()
		plt.show()

	def rollout(self):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""

		batch_obs = {}
		batch_acts = {}
		batch_log_probs = {}
		batch_rews = {}
		batch_rtgs = {}
		batch_lens = []

		# info per agent in batch (represented as an array)
		for i in range(self.num_agents): 
			batch_obs[i] = []
			batch_acts[i] = []
			batch_log_probs[i] = []
			batch_rews[i] = []
			batch_rtgs[i] = []
			# batch_lens[i] = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = {}

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		# print('ts per batch ', self.timesteps_per_batch)
		# print('ts per episode ' , self.max_timesteps_per_episode)
		while t < self.timesteps_per_batch:
			# ep_rews = [] # rewards collected per episode
			# Initialize ep_rews as an empty dictionary
			ep_rews = {i: [] for i in range(self.num_agents)}
			# print(f'initial ep_rews: {ep_rews}')

			# Reset the environment. sNote that obs is short for observation. 
			if self.env_type != 'social-dilemma': 
				obs, _ = self.env.reset() # is a list of agents, each is a dict of obs + info
			else: 
				obs = self.env.reset()

			# print(f'obs after env reset {obs}')

			# each index is val for epi for each agent 
			dones = []
			actions = []
			log_probs = []

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				actions = []
				log_probs = []

				# If render is specified, render the environment
				if self.render:
					self.env.render()

				t += 1 # Increment timesteps ran this batch so far

				# for each agent, append current obs in respective dict 
				# TODO: MAKE LESS UGLY/CLEAN
				for i in range(self.num_agents): 
					if self.env_type == 'social-dilemma': 
						# Get the observation components for the specified agent
						agent_id = f"agent-{i}"
						# print(f'curr_obs: {obs[agent_id]}')
						# print(f'obs shape \n {obs[agent_id]["curr_obs"].shape} \n {obs[agent_id]["other_agent_actions"].shape} \n {obs[agent_id]["visible_agents"].shape} \n {obs[agent_id]["prev_visible_agents"].shape}')
						
						if self.bayes: 
							agent_observation = {
								"curr_obs": obs[agent_id]["curr_obs"].flatten(),
								"other_agent_actions": obs[agent_id]["other_agent_actions"],
								"visible_agents": obs[agent_id]["visible_agents"],
								"prev_visible_agents": obs[agent_id]["prev_visible_agents"], 
								"bayes_counter": obs[agent_id]["bayes_counter"], 
							}

						else: 
							agent_observation = {
								"curr_obs": obs[agent_id]["curr_obs"].flatten(),
								"other_agent_actions": obs[agent_id]["other_agent_actions"],
								"visible_agents": obs[agent_id]["visible_agents"],
								"prev_visible_agents": obs[agent_id]["prev_visible_agents"]
							}

						# Concatenate the observation components
						concatenated_observation = np.concatenate((
							agent_observation["curr_obs"],
							agent_observation["other_agent_actions"],
							agent_observation["visible_agents"],
							agent_observation["prev_visible_agents"]
						))

						batch_obs[i].append(concatenated_observation)

					else: 
						if self.share_orientation: 
							if self.time_delay: 
								batch_obs[i].append(np.concatenate((obs[i]["agent"], obs[i]["ultrasonic"], np.array(obs[i]["neigh_orient"], obs[i]["remaining_steps"], obs[i]["neigh_remaining_steps"], obs[i]["relative_goal_pos"]))))
							else: 
								batch_obs[i].append(np.concatenate((obs[i]["agent"], obs[i]["ultrasonic"], np.array(obs[i]["neigh_orient"], obs[i]["relative_goal_pos"]))))

						else: 
							if self.time_delay: 
								# print(f'shapes of each to review: remaining steps {obs[i]["remaining_steps"]} neigh steps left: {obs[i]["neigh_remaining_steps"].shape} compared to {obs[i]["agent"]}')
								batch_obs[i].append(np.concatenate((obs[i]["agent"], obs[i]["ultrasonic"], obs[i]["remaining_steps"], obs[i]["neigh_remaining_steps"], obs[i]["relative_goal_pos"]))) 
								
							else: 
								batch_obs[i].append(np.concatenate((obs[i]["agent"], obs[i]["ultrasonic"], obs[i]["relative_goal_pos"])))


				# Calculate action and make a step in the env. 
				for i in range(self.num_agents): 

					if self.share_orientation and self.env_type != 'social-dilemma': 
						if self.time_delay: 
							action, log_prob = self.get_action(np.concatenate((obs[i]["agent"], obs[i]["ultrasonic"], obs[i]["neigh_orient"], obs[i]["remaining_steps"], obs[i]["neigh_remaining_steps"], obs[i]["relative_goal_pos"])), i)
						else: 
							action, log_prob = self.get_action(np.concatenate((obs[i]["agent"], obs[i]["ultrasonic"], obs[i]["neigh_orient"], obs[i]["relative_goal_pos"])), i)
					elif not self.share_orientation and self.env_type != 'social-dilemma': 
						if self.time_delay: 
							action, log_prob = self.get_action(np.concatenate((obs[i]["agent"], obs[i]["ultrasonic"], obs[i]["remaining_steps"], obs[i]["neigh_remaining_steps"], obs[i]["relative_goal_pos"])), i)
						else: 
							# print(f'size of relative pos: {obs[i]["relative_goal_pos"]}')
							action, log_prob = self.get_action(np.concatenate((obs[i]["agent"], obs[i]["ultrasonic"], obs[i]["relative_goal_pos"])), i)

					# TODO: need to understand how obs formatted so that we can correctly feed it to neural network 
					elif self.env_type == 'social-dilemma':
						agent_id = f"agent-{i}"

						if self.bayes: 
							agent_observation = {
							"curr_obs": obs[agent_id]["curr_obs"].flatten(),
							"other_agent_actions": obs[agent_id]["other_agent_actions"],
							"visible_agents": obs[agent_id]["visible_agents"],
							"prev_visible_agents": obs[agent_id]["prev_visible_agents"],
							"bayes_counter": obs[agent_id]["bayes_counter"] 
						}
							# Concatenate the observation components
							concatenated_observation = np.concatenate([
								agent_observation["curr_obs"],
								agent_observation["other_agent_actions"],
								agent_observation["visible_agents"],
								agent_observation["prev_visible_agents"], 
								agent_observation["bayes_counter"], 
							])


						else: 
							agent_observation = {
							"curr_obs": obs[agent_id]["curr_obs"].flatten(),
							"other_agent_actions": obs[agent_id]["other_agent_actions"],
							"visible_agents": obs[agent_id]["visible_agents"],
							"prev_visible_agents": obs[agent_id]["prev_visible_agents"],
							}

							# Concatenate the observation components
							concatenated_observation = np.concatenate([
								agent_observation["curr_obs"],
								agent_observation["other_agent_actions"],
								agent_observation["visible_agents"],
								agent_observation["prev_visible_agents"]
							])

						action, log_prob = self.get_action(concatenated_observation, i)

					# print('action', action)
					actions.append(int(action))
					log_probs.append(log_prob)


				if self.env_type == 'social-dilemma': 
					# update actions to be compatible with map_env
					act = {}
					for i in range(self.num_agents): 
						agent_id = f"agent-{i}"
						act[agent_id] = actions[i]

					actions = act

					obs, rews, dones, _ = self.env.step(actions)

				else:
					obs, rews, dones, _, _ = self.env.step(actions)

				# print(f'info collected so far: \n obs {obs} \n rews {rews} \n log_probs {log_probs} \n ep_rews {ep_rews}')

				# Track recent reward, action, and action log probability
				for i in range(self.num_agents): 
					agent_id = f"agent-{i}"
					# for each agent dict, add rews collected for that episode 

					if self.env_type == 'social-dilemma': 
						ep_rews[i].append(rews[agent_id])
						batch_acts[i].append(actions[agent_id])
						batch_log_probs[i].append(log_probs[i].item())

					else: 

						ep_rews[i].append(rews[i])
						# batch_rews[i].extend(ep_rews[i])

						batch_acts[i].append(actions[i])
						# print(f'log info : {log_probs[i].item()}')
						batch_log_probs[i].append(log_probs[i].item())

					# dones.append(done)

				# If the environment tells us the episode is terminated, break
				if dones:
					break

			# print(f'info from batch ep_rews: {ep_rews} \n batch_acts: {batch_acts} \n batch_log_probs: {batch_log_probs} \n batch_obs: {batch_obs}')

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)

			for i in range(self.num_agents): 
				batch_rews[i].append(ep_rews[i])

		# Reshape data as tensors in the shape specified in function description, before returning
		# batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		# batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		# batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()
		# batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

		# convert each obs per agent into tensor 
		for i in range(self.num_agents):

			batch_obs[i] = torch.tensor(batch_obs[i], dtype=torch.float)
			batch_acts[i] = torch.tensor(batch_acts[i], dtype=torch.float)
			batch_log_probs[i] = torch.tensor(batch_log_probs[i], dtype=torch.float).flatten()
			batch_rtgs[i] = self.compute_rtgs(batch_rews[i])

		# Now, batch_obs, batch_acts, and batch_log_probs are dictionaries with tensor values
		# print(f'size of batch_rews: {len(batch_rews[0])}, {len(batch_rews[0][0])}')
		# print(f'batch rews: {batch_rews}')

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num episodes per batch, num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs
		
	def _update_live_plot(self, ax, line, x, y, iteration, agent_id, csv_filename):
		if not line:
			# If the line does not exist, create a new line
			line, = ax.plot([], [], label='Total Average Episodic Return')
			ax.legend()
		# Assuming line is a Line2D object
		line.set_xdata(np.append(line.get_xdata(), iteration))
		line.set_ydata(np.append(line.get_ydata(), y))
		ax.relim()  # Recalculate limits
		ax.autoscale_view()  # Autoscale the view
		ax.figure.canvas.draw()

		
		# Save data to CSV
		with open(csv_filename, 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			if agent_id != 'total': 
				writer.writerow([iteration, agent_id, y])
			else: 
				writer.writerow([iteration, y])



	def get_action(self, obs, index):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for a mean action
		# mean = self.actor(obs)
		# print(f'obs shape: {obs.shape}')
		mean = self.actors[index](obs)
		dist = torch.distributions.Categorical(logits=mean)

		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		# dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# If we're testing, just return the deterministic action. Sampling should only be for training
		# as our "exploration" factor.
		if self.deterministic:
			return mean.detach().numpy(), 1

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts, batch_rtgs, agent_id):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)
				batch_rtgs - the rewards-to-go calculated in the most recently collected
								batch as a tensor. Shape: (number of timesteps in batch)
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critics[agent_id](batch_obs).squeeze()
		# V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		# mean = self.actor(batch_obs)
		mean = self.actors[agent_id](batch_obs)
		# dist = MultivariateNormal(mean, self.cov_mat)
		dist = torch.distributions.Categorical(logits=mean)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = False                             # If we should render during rollout
		self.save_freq = 10                             # How often we save in number of iterations
		self.deterministic = False                      # If we're testing, don't sample actions
		self.seed = None								# Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])

		# avg_ep_rews = np.mean([np.sum(reward_list) for reward_list in self.logger['batch_rews'].values()])
		avg_ep_rews = np.mean([np.sum(episode_rewards) for episode_rewards in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []