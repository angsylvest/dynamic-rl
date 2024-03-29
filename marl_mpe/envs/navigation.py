import numpy as np
import pygame

import gym
from gym import spaces
import matplotlib.pyplot as plt
import random 


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, num_agents = 1, obs_type = "simple pos", time_delay = False, nonholonomic = False, gifting = False):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        self.num_agents = num_agents
        self.obs_type = obs_type
        self.introduce_time_delay = True # time_delay # assume this is norm s
        self.nonholonomic = nonholonomic
        self.gifting = gifting

        # coordination-specific info (incorporates flocking vs compliment actions in action space)
        self.nature_based = False 

        # reward info
        self.collision_radius_threshold = 1.0
        self.collision_penalty = 1.0
        self.proximity_reward = 0.5
        self.goal_reward = 1.0
        self.proximity_threshold = 5.0

        # Generate dark random colors for each agent
        # TODO: make so this doesn't need to be hardcoded
        self.agent_colors = [(0, 0, 255), (255, 0, 0)]
        
        if not self.gifting and not self.nature_based: 
            self.action_space = [
                spaces.Discrete(4) for _ in range(self.num_agents)
            ]
        else: 
            self.action_space = [
                spaces.Discrete(5) for _ in range(self.num_agents)
            ]

        self.observation_space = [
            spaces.Dict(
                {
                    "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "remaining_steps": spaces.Box(0, size - 1, shape=(1,), dtype=int),
                    "orientation": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "ultrasonic": spaces.Box(0, size - 1, shape=(4,), dtype=int),
                    "neigh_orient": spaces.Box(0, size - 1, shape=(4,), dtype=int),
                    "neigh_remaining_steps": spaces.Box(0, size - 1, shape=(4,), dtype=int),
                    "relative_goal_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "other_agent_act": spaces.Box(0, size - 1, shape=(num_agents,), dtype=int),
                }
        ) for _ in range(self.num_agents)
        ]

        self.agent_obs_info = [
            {"agent": np.array([0,0]), 
             "target": np.array([0,0]),
             "remaining_steps": np.array([-1]), 
             "orientation": np.array([0,0]),
             "ultrasonic": np.array([0,0,0,0]), # N, W, E, S 
             "neigh_orient": np.array([0,0,0,0]), # ←, ↑, →, ↓ (1, 2, 3, 4) else 0 if no agent
             "neigh_remaining_steps": np.array([0,0,0,0]), # incorporate speed heterogeneity
             "other_agent_act": np.zeros(num_agents)
             }

             for i in range(self.num_agents)

        ]

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0,0]) # waiting (potential gifter)
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def update_ultrasonic(self):
        for idx, agent_info in enumerate(self.agent_obs_info):
            cur_pos = np.array(agent_info["agent"])

            # Define orientations and positions relative to the current agent
            orientations = np.array([(0, 1), (-1, 0), (1, 0), (0, -1)])
            neighbor_positions = cur_pos + orientations

            # Convert neighbor positions to a set for faster lookup
            neighbor_positions_set = set(map(tuple, neighbor_positions))

            # Find occupied positions and neighbor orientations
            occupancy = np.zeros(4, dtype=int)
            neigh_orientation = np.full(4, -1, dtype=int)  # Initialize with -1 for unmatched orientations
            neigh_timestep = np.full(4, -1, dtype=int)  # Initialize with -1 for unmatched orientations

            # Find matching positions and orientations
            for agent_id, other_agent_info in enumerate(self.agent_obs_info):
                if tuple(other_agent_info["agent"]) in neighbor_positions_set:
                    pos_index = np.where(np.all(neighbor_positions == other_agent_info["agent"], axis=1))[0][0]
                    occupancy[pos_index] = 1
                    neigh_orientation[pos_index] = agent_id  # Store the index of the agent
                    neigh_timestep[pos_index] = other_agent_info["remaining_steps"]

            # Update agent observation info
            agent_info["ultrasonic"] = occupancy
            agent_info["neigh_orient"] = neigh_orientation
            agent_info["neigh_remaining_steps"] = neigh_timestep

    def _get_obs(self, index=None):
        self.update_ultrasonic()  # TODO: maybe need to fix to be more efficient 

        if index is not None:
            obs = {
                "agent": self.agent_obs_info[index]["agent"],
                "target": self.agent_obs_info[index]["target"],
                "ultrasonic": self.agent_obs_info[index]["ultrasonic"],
                "neigh_orient": self.agent_obs_info[index]["neigh_orient"]
            }
            # Calculate relative goal position
            agent_pos = self.agent_obs_info[index]["agent"]
            goal_pos = self.agent_obs_info[index]["target"]
            relative_goal_pos = goal_pos - agent_pos
            obs["relative_goal_pos"] = np.array(relative_goal_pos)
            # print(relative_goal_pos)

            if self.introduce_time_delay:
                obs["remaining_steps"] = self.agent_obs_info[index]["remaining_steps"]
                obs["neigh_remaining_steps"] = self.agent_obs_info[index]["neigh_remaining_steps"]

            return obs
        else:
            obs_list = []
            for i in range(self.num_agents):
                obs = {
                    "agent": self.agent_obs_info[i]["agent"],
                    "target": self.agent_obs_info[i]["target"],
                    "ultrasonic": self.agent_obs_info[i]["ultrasonic"],
                    "neigh_orient": self.agent_obs_info[i]["neigh_orient"]
                }
                # Calculate relative goal position
                agent_pos = self.agent_obs_info[i]["agent"]
                goal_pos = self.agent_obs_info[i]["target"]
                relative_goal_pos = goal_pos - agent_pos
                obs["relative_goal_pos"] = np.array(relative_goal_pos)

                if self.introduce_time_delay:
                    obs["remaining_steps"] = self.agent_obs_info[i]["remaining_steps"]
                    obs["neigh_remaining_steps"] = self.agent_obs_info[i]["neigh_remaining_steps"]

                obs_list.append(obs)
            return obs_list

    def _get_info(self, index = None):
        if index is not None: 
            return {
                "distance": np.linalg.norm(
                    self.agent_obs_info[index]["agent"] - self.agent_obs_info[index]["target"], ord=1
                )
            }
        else: 
            return [{
                "distance": np.linalg.norm(
                    self.agent_obs_info[i]["agent"] - self.agent_obs_info[i]["target"], ord=1
                )
            } 
            
            for i in range(self.num_agents)]


    def reset(self, seed=None, options=None): # TODO: allow seed to be updateable
        # We need the following line to seed self.np_random
        self.np_random = np.random.RandomState(seed)


        for i in range(self.num_agents): 
            # Choose the agent's location uniformly at random
            agent_location = self.np_random.randint(0, self.size, size=2, dtype=int)

            # We will sample the target's location randomly until it does not coincide with the agent's location
            # Also will check to make sure locations don't overlap with others 

            target_location = agent_location
            while np.array_equal(target_location, agent_location):
                target_location = self.np_random.randint(
                    0, self.size, size=2, dtype=int
                )

            self.agent_obs_info[i]["agent"] = agent_location
            self.agent_obs_info[i]["target"] = target_location

            # random orientation (choice between (-1,0),(1,0),(0,1),(0,-1))
            orientations = [(-1, 0), (1, 0), (0, 1), (0, -1)]
            random_orientation = np.array(random.choice(orientations))

            self.agent_obs_info[i]["orientation"] = random_orientation
            self.agent_obs_info[i]["remaining_steps"] = np.array([0])


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info # by default will return all data 
    
    def reward(self, agent_id): 
        current_loc = np.array(self.agent_obs_info[agent_id]["agent"])
        target = np.array(self.agent_obs_info[agent_id]["target"])

        # Calculate distance to the goal
        distance_to_goal = np.linalg.norm(current_loc - target)
        distance_to_goal = np.clip(distance_to_goal, a_min=1e-10, a_max=None)

        # Check if the agent is on the goal
        on_goal = distance_to_goal < 1e-5  # You might need to adjust this threshold

        # Check for collisions with other agents
        collision_penalty_sum = 0.0

        # Calculate a proximity-based reward component
        proximity_reward = np.exp(-distance_to_goal / self.proximity_threshold)

        for i in range(self.num_agents):
            if i != agent_id:
                other_agent_loc = np.array(self.agent_obs_info[i]["agent"])
                distance_to_other_agent = np.linalg.norm(current_loc - other_agent_loc)

                # If the distance is below a threshold, penalize for collision
                if distance_to_other_agent < self.collision_radius_threshold:
                    collision_penalty_sum += self.collision_penalty


        # Calculate the total reward
        total_reward = self.goal_reward - collision_penalty_sum + proximity_reward if on_goal else -collision_penalty_sum + proximity_reward
        return total_reward


    def validate_gifting(self, actions):
        gift_neighb = {}

        if 5 not in actions:
            return False, "NA" 

        for agent_id, action in enumerate(actions): 
            curr_loc = self.agent_obs_info[agent_id]["agent"] 
            if action == 5: 
                for pos in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor_loc = (curr_loc[0] + pos[0], curr_loc[1] + pos[1])
                    for a in range(self.num_agents):
                        if neighbor_loc == self.agent_obs_info[a]["agent"]:
                            gift_neighb[agent_id] = neighbor_loc
                            break
                    
                # if there is no neighbor, want to penalize 
                # also want to penalize if both neighboring agent's choose to defer
                

        return True, gift_neighb  # Add appropriate return value


    def step(self, actions):
        # updated to handle non-holonomic movements, won't instanteously move to loc
    
        rewards = []
        observations = []
        terminated = []
        infos = []

        # print(f'actions: {actions}')
        # if gifting, first want to verify reward from nearest agent
        # check = False # initially false until validated 
        # if self.gifting: 
        #     check, list = self.validate_gifting(actions)
        #     print(f'gifting output: {list}')
            
        agent_id_dict = {0: 1, 1:0}
        leader_dict = {0: "leader", 1: "follower"}
        for agent_id, action in enumerate(actions):
            reward_penalty = 0

            if not self.introduce_time_delay or ((self.agent_obs_info[agent_id]["remaining_steps"] < 0) and self.introduce_time_delay):
                act_agent = action 

                if self.nature_based: 
                    if action == 4: # mimic neighboring agent 
                       # TODO: need way to handle if they are both in state 
                        other_act = agent_id_dict[agent_id]
                        act_agent = other_act 

                        if other_act == 4 and leader_dict[agent_id] == "leader": 
                           # need one to take leader role, penalize trying to copy  
                            reward_penalty = -1.2 
                        elif other_act == 5 and leader_dict[agent_id] == "follower": 
                            # get some reward from agent for demuring
                            reward_penalty = 0.5

                    

                direction = self._action_to_direction[act_agent]
                # print(f'self.agent_obs_info: {self.agent_obs_info} for agent_id {agent_id}')
                loc = self.agent_obs_info[agent_id]["agent"]


                if self.nonholonomic and (self.agent_obs_info[agent_id]["orientation"] != direction).any(): # will move vector unless it's pointing in same direction
                    
                    self.agent_obs_info[agent_id]["orientation"] = direction 

                else: 
                    # Calculate new x and y values
                    new_x = loc[0] + direction[0]
                    new_y = loc[1] + direction[1]

                    # Clip the values to be within the desired range
                    clipped_x = np.clip(new_x, 0, self.size - 1)
                    clipped_y = np.clip(new_y, 0, self.size - 1)

                    # Update the observation_space
                    self.agent_obs_info[agent_id]["agent"] = np.array([clipped_x, clipped_y])

                    if self.introduce_time_delay:
                        # add time delay once given new action before moving on to next action
                        self.agent_obs_info[agent_id]["remaining_steps"] = np.array([1]) if agent_id == 0 else np.array([2])
                        # print(f'updating obs to {self.agent_obs_info[agent_id]["agent"]} with updated remaining steps')

            else: 
                # agent must remain at current state 
                self.agent_obs_info[agent_id]["remaining_steps"] -= 1
                # print(f'decrementing steps, current state {self.agent_obs_info[agent_id]["remaining_steps"] }')
            
            # TODO: hard-coded, must make more logical 
            # if check and actions[agent_id] == 4: 
            #     if list[agent_id] != "NA": 
            #         current_reward = 0.25 # retroactive reward 
            #         print(f'retroactive reward granted ')
            #     else: 
            #         current_reward = self.reward(agent_id)*0.25
            # else: 
                
            current_reward = self.reward(agent_id) + reward_penalty # TODO: update reward function

            terminated_agent = np.array_equal(
                self.agent_obs_info[agent_id]["agent"] , self.agent_obs_info[agent_id]["target"]
            )

            terminated.append(terminated_agent)
            # reward = 1 if terminated_agent else 0 # simple reward, doesn't use ttc or collision 
            rewards.append(current_reward)
            observation = self._get_obs(agent_id)
            observations.append(observation)
            info = self._get_info()
            infos.append(info)

        if self.render_mode == "human":
            self._render_frame()

        # Assuming your environment considers an episode terminated if any agent reaches the target
        episode_terminated = all(terminated)

        return observations, rewards, episode_terminated, {}, infos
    


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw the targets and agents using unique colors for each agent
        for i in range(self.num_agents):
            target_color = self.agent_colors[i]
            agent_color = self.agent_colors[i]

            pygame.draw.rect(
                canvas,
                target_color,
                pygame.Rect(
                    pix_square_size * self.agent_obs_info[i]["target"],
                    (pix_square_size, pix_square_size),
                ),
            )

            pygame.draw.circle(
                canvas,
                agent_color,
                (self.agent_obs_info[i]["agent"] + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

            if self.nonholonomic: 
                    
                # Get the orientation vector
                orientation_vector = np.array(self.agent_obs_info[i]["orientation"])

                # Calculate the endpoint of the orientation vector
                endpoint = (
                    int((self.agent_obs_info[i]["agent"][0] + 0.5) * pix_square_size + orientation_vector[0] * pix_square_size),
                    int((self.agent_obs_info[i]["agent"][1] + 0.5) * pix_square_size + orientation_vector[1] * pix_square_size)
                )

                # Draw the orientation vector
                pygame.draw.line(
                    canvas,
                    agent_color, 
                    (int((self.agent_obs_info[i]["agent"][0] + 0.5) * pix_square_size), int((self.agent_obs_info[i]["agent"][1] + 0.5) * pix_square_size)),
                    endpoint,
                    width=3,
                )

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



def test_render(): 
    env = GridWorldEnv(render_mode = "rgb_array", num_agents=2, time_delay=True, nonholonomic=False)
    print('env created')
    obs = env.reset()
    print(env.step(actions=[0, 4]))
    print('obs reset')
    screen = env.render()

    # Convert the screen array to a numeric type
    screen = screen.astype(float)

    plt.imshow(screen)
    print('showing')
    plt.pause(20)


def main():
    test_render()

# main()
            