import random
import numpy as np


use_thompson = False
decay = True 
invert = True

class NArmedBandit:

    def __init__(self, probs):
        self.probs = np.array(probs)
        assert np.all(self.probs >= 0)
        assert np.all(self.probs <= 1)

        self.optimal_reward = np.max(self.probs)
        self.n_arm = len(self.probs)

    def get_observation(self):
        return self.n_arm
    
    def get_optimal_reward(self):
        return np.max(self.probs)
    
    def get_expected_reward(self, action):
        return self.probs[action]
    
    def get_stochastic_reward(self, action):
        return int(random.random() < self.probs[action])

class NArmedBanditDrift(NArmedBandit):
    def __init__(self, n_arm, a0=1., b0=1., gamma=0.01):  # self.gamma = 0 means no drift
        self.n_arm = n_arm 
        self.a0 = a0
        self.b0 = b0 

        self.prior_success = np.array([a0 for _ in range(n_arm)])
        self.prior_failure = np.array([b0 for _ in range(n_arm)])

        self.gamma = gamma 
        self.probs = np.array([random.betavariate(a0, b0) for _ in range(n_arm)])

    def decay_gamma(self, decay_rate = 0.95):
        self.gamma = max(self.gamma * decay_rate, 0.001)  # Ensure gamma doesn't become too small

    def set_prior(self, prior_success, prior_failure):
        self.prior_success = prior_success
        self.prior_failure = prior_failure

    def get_optimal_reward(self):
        return np.max(self.probs)
    
    def advance(self, action, reward):

        if decay: 
            self.decay_gamma()

        self.prior_success = self.prior_success * (1 - self.gamma) + self.a0 * self.gamma
        self.prior_failure = self.prior_failure * (1 - self.gamma) + self.b0 * self.gamma

        self.prior_success[action] += reward
        self.prior_failure[action] += 1 - reward
        
        # Ensure that prior_success and prior_failure are always greater than 0.0
        self.prior_success = np.maximum(self.prior_success, 0.001)
        self.prior_failure = np.maximum(self.prior_failure, 0.001)

        # resample posterior probabilities 
        self.probs = np.array([random.betavariate(self.prior_success[a], self.prior_failure[a]) for a in range(self.n_arm)])


    def sample_action(self):
        if use_thompson:
            # Use Thompson Sampling
            sampled_theta = np.array([random.betavariate(self.prior_success[a], self.prior_failure[a]) for a in range(self.n_arm)])
            return np.argmax(sampled_theta)
        
        else: 
            # use categorical sampling 
            if invert: 
                reciprocal_probs = 1 / self.probs
                new_probs = reciprocal_probs / np.sum(reciprocal_probs)

                return np.random.choice(range(self.n_arm), p=(new_probs))
            else: 
                return np.random.choice(range(self.n_arm), p=self.probs)
