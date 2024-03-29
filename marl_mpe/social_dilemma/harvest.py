import numpy as np
from numpy.random import rand

from agent import HarvestAgent
from agent_follower import HarvestAgentFollower
from discrete_with_d_type import DiscreteWithDType
from map_env import MapEnv
from maps import HARVEST_MAP

import globals as globals

APPLE_RADIUS = 2

# Add custom actions to the agent

if globals.gifting: 
    _HARVEST_ACTIONS = {"FIRE": 5}  # length of firing range
else: 
    _HARVEST_ACTIONS = {}  # empty for now


SPAWN_PROB = [0, 0.005, 0.02, 0.05]

HARVEST_VIEW_SIZE = 7


class HarvestEnv(MapEnv):
    def __init__(
        self,
        ascii_map=HARVEST_MAP,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0,
        split_roles = False 
    ):
        self.split_roles = split_roles
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

        # determine what type of env being used 
        self.gifting = globals.gifting


    
    # updated to remove FIRE BEAM action
    @property
    def action_space(self):
        if self.gifting: # include "gift" action
            return DiscreteWithDType(8, dtype=np.uint8)
        else: 
            return DiscreteWithDType(7, dtype=np.uint8)
    
    @property
    def action_space_roles(self):
        return DiscreteWithDType(10, dtype=np.uint8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            if i % 2 == 0 or not self.split_roles: 
                agent_id = "agent-" + str(i)
                spawn_point = self.spawn_point()
                rotation = self.spawn_rotation()
                grid = map_with_agents
                agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
                self.agents[agent_id] = agent

            else: 
                agent_id = "agent-" + str(i)
                spawn_point = self.spawn_point()
                rotation = self.spawn_rotation()
                grid = map_with_agents
                agent = HarvestAgentFollower(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
                self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")


    def custom_action(self, agent, action):
        # TODO: ensure that this is actually doing what is expected if gifting 

        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE" and self.gifting:
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
                agent=agent.agent_id
            )

        return updates
            

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if (
                                0 <= x + j < self.world_map.shape[0]
                                and self.world_map.shape[1] > y + k >= 0
                            ):
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples
    
    def render_current_map(self): 
        pass 

    

def main(): # testing render here 
    env = HarvestEnv(num_agents=3,
                return_agent_actions=True,
                use_collective_reward=True,
                inequity_averse_reward=True,
                alpha=0.0,
                beta=0.0)
    
    # env.custom_reset() # will generate apples 
    # env.custom_map_update()
    states = env.reset()
    obs = states
    print(f'obs shape \n {obs["agent-0"]["curr_obs"].shape} \n {obs["agent-0"]["other_agent_actions"].shape} \n {obs["agent-0"]["visible_agents"].shape} \n {obs["agent-0"]["prev_visible_agents"].shape}')

    env.render()
     


# main()