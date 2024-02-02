import numpy as np
import pygame

import gym
from gym import spaces
import matplotlib.pyplot as plt


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, num_agents = 1, obs_type = "simple pos"):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        self.num_agents = num_agents
        self.obs_type = obs_type
        self.introduce_time_delay = True

        # reward info
        self.collision_radius_threshold = 2.0
        self.collision_penalty = 1.0
        self.proximity_reward = 0.5
        self.goal_reward = 1.0

        self.action_space = [
            spaces.Discrete(4) for _ in range(self.num_agents)
        ]

        self.observation_space = [
            spaces.Dict(
                {
                    "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "remaining_steps": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                }
        ) for _ in range(self.num_agents)
        ]

        self.agent_obs_info = [
            {"agent": np.array([0,0]), 
             "target": np.array([0,0]),
             "remaining_steps": np.array([-1])
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

    def _get_obs(self, index = None):
        if index is not None: 
            return {"agent": self.agent_obs_info[index]["agent"], "target": self.agent_obs_info[index]["target"]}
        else: 
            return [
                {"agent": self.agent_obs_info[i]["agent"], "target": self.agent_obs_info[i]["target"]}
                for i in range(self.num_agents) 
            ]

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

        # Check for collisions with other agents
        collision_penalty_sum = 0.0

        for i in range(self.num_agents):
            if i != agent_id:
                other_agent_loc = np.array(self.agent_obs_info[i]["agent"])
                distance_to_other_agent = np.linalg.norm(current_loc - other_agent_loc)

                # If the distance is below a threshold, penalize for collision
                if distance_to_other_agent < self.collision_radius_threshold:
                    collision_penalty_sum += self.collision_penalty

        # Calculate the total reward
        total_reward = self.goal_reward - collision_penalty_sum + self.proximity_reward / (distance_to_goal + 1e-6)

        return total_reward

        


    def step(self, actions):
    
        rewards = []
        observations = []
        terminated = []
        infos = []

        # print(f'actions: {actions}')

        for agent_id, action in enumerate(actions):

            if not self.introduce_time_delay or ((self.agent_obs_info[agent_id]["remaining_steps"] < 0) and self.introduce_time_delay):
 
                direction = self._action_to_direction[action]
                # print(f'self.agent_obs_info: {self.agent_obs_info} for agent_id {agent_id}')
                loc = self.agent_obs_info[agent_id]["agent"]

                # Calculate new x and y values
                new_x = loc[0] + direction[0]
                new_y = loc[1] + direction[1]

                # Clip the values to be within the desired range
                clipped_x = np.clip(new_x, 0, self.size - 1)
                clipped_y = np.clip(new_y, 0, self.size - 1)

                # Update the observation_space
                self.agent_obs_info[agent_id]["agent"] = np.array([clipped_x, clipped_y])

                # current_reward = self.reward(agent_id) # TODO: update reward function

                # terminated_agent = np.array_equal(
                #     self.agent_obs_info[agent_id]["agent"] , self.agent_obs_info[agent_id]["target"]
                # )

                if self.introduce_time_delay:
                    # add time delay once given new action before moving on to next action
                    self.agent_obs_info[agent_id]["remaining_steps"] = 2
                    # print(f'updating obs to {self.agent_obs_info[agent_id]["agent"]} with updated remaining steps')

            else: 
                # agent must remain at current state 
                self.agent_obs_info[agent_id]["remaining_steps"] -= 1
                # print(f'decrementing steps, current state {self.agent_obs_info[agent_id]["remaining_steps"] }')
            

            current_reward = self.reward(agent_id) # TODO: update reward function

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


        # First we draw the target(s)
        for i in range(self.num_agents): 
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * self.agent_obs_info[i]["target"],
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent(s)
        for i in range(self.num_agents): 
            pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent_obs_info[i]["agent"] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
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
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
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
    env = GridWorldEnv(render_mode = "rgb_array", num_agents=2)
    print('env created')
    obs = env.reset()
    print(env.step(actions=[0, 1]))
    print('obs reset')
    screen = env.render()

    # Convert the screen array to a numeric type
    screen = screen.astype(float)

    plt.imshow(screen)
    print('showing')
    plt.pause(20)

# def main():
#     test_render()

# main()
            