"""Base class for an agent that defines the possible actions. """

import numpy as np

import utility_funcs as util
from bayes import NArmedBanditDrift
import globals as globals

# basic moves every agent should do
BASE_ACTIONS = {
    0: "MOVE_LEFT",  # Move left
    1: "MOVE_RIGHT",  # Move right
    2: "MOVE_UP",  # Move up
    3: "MOVE_DOWN",  # Move down
    4: "STAY",  # don't move
    5: "TURN_CLOCKWISE",  # Rotate counter clockwise
    6: "TURN_COUNTERCLOCKWISE",
}  # Rotate clockwise


class Agent(object):
    def __init__(self, agent_id, start_pos, start_orientation, full_map, row_size, col_size):
        """Superclass for all agents.

        Parameters
        ----------
        agent_id: (str)
            a unique id allowing the map to identify the agents
        start_pos: (np.ndarray)
            a 2d array indicating the x-y position of the agents
        start_orientation: (np.ndarray)
            a 2d array containing a unit vector indicating the agent direction
        full_map: (2d array)
            a reference to this agent's view of the environment
        row_size: (int)
            how many rows up and down the agent can look
        col_size: (int)
            how many columns left and right the agent can look
        """
        self.agent_id = agent_id
        self.pos = np.array(start_pos)
        self.orientation = start_orientation
        self.full_map = full_map
        self.row_size = row_size
        self.col_size = col_size
        self.reward_this_turn = 0
        self.prev_visible_agents = None

        # bayes intuition here 
        # if still in delay mode, will get penalized 
        self.max_delay = globals.max_delay
        self.bayes = NArmedBanditDrift(n_arm=self.max_delay)
        self.curr_restraint = 0 
        self.using_bayes = globals.bayes
        self.consume_reward = 1 # normally 1 but would be scaled if practicing restraint

        self.gifting = globals.gifting
        self.accrued_debt = 0

        self.agent_perf = {'num_collected': 0, 'time_waited': 0, 'num_cleaned': 0}

        assert self.gifting != self.bayes # ensure that they are not being used together
    
    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete, or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        raise NotImplementedError

    def get_char_id(self):
        return bytes(str(int(self.agent_id[-1]) + 1), encoding="ascii")

    def get_state(self):
        return util.return_view(self.full_map, self.pos, self.row_size, self.col_size)

    def compute_reward(self, reset = True):
        reward = self.reward_this_turn

        if reset: 
            self.reward_this_turn = 0

            if globals.bayes: 
                self.accrued_debt = 0 
                self.consume_reward = 1

        self.agent_perf = {'num_collected': 0, 'time_waited': 0, 'num_cleaned': 0}

        return reward

    def set_pos(self, new_pos):
        self.pos = np.array(new_pos)

    def get_pos(self):
        return self.pos

    def translate_pos_to_egocentric_coord(self, pos):
        offset_pos = pos - self.pos
        ego_centre = [self.row_size, self.col_size]
        return ego_centre + offset_pos

    def set_orientation(self, new_orientation):
        self.orientation = new_orientation

    def get_orientation(self):
        return self.orientation
    
    def reset_metrics(self): 
        self.agent_perf = {'num_collected': 0, 'time_waited': 0}

    def return_valid_pos(self, new_pos):
        """Checks that the next pos is legal, if not return current pos"""
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # You can't walk through walls, closed doors or switches
        if self.is_tile_walkable(new_row, new_col):
            return new_pos
        else:
            return self.pos

    def update_agent_pos(self, new_pos):
        """Updates the agents internal positions

        Returns
        -------
        old_pos: (np.ndarray)
            2 element array describing where the agent used to be
        new_pos: (np.ndarray)
            2 element array describing the agent positions
        """
        old_pos = self.pos
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        if self.is_tile_walkable(new_row, new_col):
            validated_new_pos = new_pos
        else:
            validated_new_pos = self.pos
        self.set_pos(validated_new_pos)
        # TODO(ev) list array consistency
        return self.pos, np.array(old_pos)

    def is_tile_walkable(self, row, column):
        return (
            0 <= row < self.full_map.shape[0]
            and 0 <= column < self.full_map.shape[1]
            # You can't walk through walls, closed doors or switches
            and self.full_map[row, column] not in [b"@", b"D", b"w", b"W"]
        )

    def update_agent_rot(self, new_rot):
        self.set_orientation(new_rot)

    def hit(self, char):
        """Defines how an agent responds to being hit by a beam of type char"""
        raise NotImplementedError

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        raise NotImplementedError


HARVEST_ACTIONS = BASE_ACTIONS.copy()

if globals.gifting: 
    HARVEST_ACTIONS.update({7: "FIRE"})  # Fire a penalty beam


class HarvestAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, full_map, view_len):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, full_map, view_len, view_len)
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)
        

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return HARVEST_ACTIONS[action_number]

    def hit(self, char, split_cost = 0):
        if char == b"F" and not self.gifting:
            self.reward_this_turn -= 50
        elif char == b"F" and self.gifting: 
            self.reward_this_turn += split_cost 

    def fire_beam(self, char):
        if char == b"F":
            self.reward_this_turn -= 1


    def get_done(self):
        return False
    

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if self.using_bayes: 
            if char == b"A":
                self.agent_perf['num_collected'] += 1
                self.reward_this_turn += self.consume_reward
                return b" "

                # if self.curr_restraint > 0: 
                #     self.consume_reward *= 0.25 # only get quarter of current reward
                #     self.reward_this_turn += self.consume_reward # 0.5 # less gain if acting against restraint
                #     # print(f'updating reward {self.reward_this_turn} is self.curr_restraint is greater than or equal to 0 {self.curr_restraint} for {self.reward_this_turn}')
                #     return b""
                # else: 
                #     self.consume_reward = 1
                #     self.reward_this_turn += self.consume_reward
                #     # print(f'updating reward {self.reward_this_turn} is self.curr_restraint is less than 0 {self.curr_restraint} for {self.reward_this_turn}')
                #     # self.bayes.advance(self, self.curr_restraint, self.reward_this_turn)

                #     # self.curr_restraint = self.bayes.sample_action()
                #     return b" "
            else: 
                # no item to consume 
                self.agent_perf['time_waited'] += 1

                if self.curr_restraint > 0 and self.using_bayes: 
                    if self.consume_reward < 1: 
                        self.consume_reward *= 1.25 # some return back 
                        # self.reward_this_turn += self.consume_reward

                return char

        else: 
            # print('defaulting to other ')
            if char == b"A":
                self.agent_perf['num_collected'] += 1
                self.reward_this_turn += 1
                return b" "
            else:
                self.agent_perf['time_waited'] += 1
                return char


CLEANUP_ACTIONS = BASE_ACTIONS.copy()

if globals.gifting: 
    CLEANUP_ACTIONS.update({7: "FIRE", 8: "CLEAN"})  # Fire a penalty beam  # Fire a cleaning beam
else: 
    CLEANUP_ACTIONS.update({7: "CLEAN"})  

class CleanupAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, full_map, view_len):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, full_map, view_len, view_len)
        # remember what you've stepped on
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)
        self.cleaned = False 


    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return CLEANUP_ACTIONS[action_number]

    def fire_beam(self, char, updates = []):
        if char == b"F":
            self.reward_this_turn -= 1
            print(f'fire beaming reward this turn: {self.reward_this_turn}')

        if char == b"C":
            if updates != []:
                self.reward_this_turn += 0.01
                self.cleaned = True 
        #     self.agent_perf['num_cleaned'] += 1 
        #     # print(f'cleaning up the env')

        #     if self.bayes: 
        #         self.curr_restraint -= 1
        #         # if cleaning will get temporary reward (from someone else)
        #         # if self.consume_reward > 0: 
        #             # self.reward_this_turn += 0.1
        #             # self.accrued_debt += 1
        #             # self.consume_reward -= 0.1 # hopefully will induce heterogeneity
                    
        #         # self.curr_restraint = 0


    def get_done(self):
        return False

    def hit(self, char, split_cost = 0):
        if char == b"F" and not self.gifting:
            self.reward_this_turn -= 50
        elif char == b"F" and self.gifting: 
            self.reward_this_turn += split_cost

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        # """Defines how an agent interacts with the char it is standing on"""
        if self.using_bayes: 
            if char == b"A":
                self.agent_perf['num_collected'] += 1
                self.reward_this_turn += self.consume_reward
                print(f'consuming an apple')
                self.cleaned = False
                return b""

                # # get amount of time waited 
                # if self.curr_restraint > 0: 
                #     self.consume_reward *= 0.9
                #     self.reward_this_turn += self.consume_reward # 0.5 # less gain if acting against restraint
                #     # print(f'updating reward is self.curr_restraint is greater than or equal to 0 {self.curr_restraint} for {self.reward_this_turn}')
                #     return b""
                # else: 
                #     self.consume_reward = 1
                #     self.reward_this_turn += self.consume_reward
                #     # print(f'updating reward is self.curr_restraint is less than 0 {self.curr_restraint} for {self.reward_this_turn}')
                #     # self.bayes.advance(self, self.curr_restraint, self.reward_this_turn)

                #     self.curr_restraint = self.bayes.sample_action()
                #     return b" "

            else:
                # want to reward for how close to nearest apple
                if not self.cleaned: 
                    self.reward_this_turn -= 0.5
                self.reward_this_turn -= 0.5
                self.agent_perf['time_waited'] += 1
                self.cleaned = False 
                return char

        else: 
            # print(f'defaulting to other {self.reward_this_turn} with char {char}')
            if char == b"A":
                self.agent_perf['num_collected'] += 1
                self.reward_this_turn += 1
                print(f'yum yum apples')
                self.cleaned = False
                return b" "
            else:
                # want to reward for how close to nearest apple
                if not self.cleaned: 
                    self.reward_this_turn -= 0.5
                self.agent_perf['time_waited'] += 1
                self.cleaned = False
                return char


SWITCH_ACTIONS = BASE_ACTIONS.copy()
SWITCH_ACTIONS.update({7: "TOGGLE_SWITCH"})  # Fire a switch beam


class SwitchAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, full_map, view_len):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, full_map, view_len, view_len)
        # remember what you've stepped on
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)
        self.is_done = False

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return SWITCH_ACTIONS[action_number]

    def fire_beam(self, char):
        # Cost of firing a switch beam
        # Nothing for now.
        if char == b"F":
            self.reward_this_turn += 0

    def get_done(self):
        return self.is_done

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == b"d":
            self.reward_this_turn += 1
            self.is_done = True
            return b" "
        else:
            return char