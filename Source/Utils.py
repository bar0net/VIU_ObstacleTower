# -*- coding: utf-8 -*-
"""
@author: Jordi Tudela Alcacer
"""
from collections import deque
import numpy as np
import copy
import os

import Source.Models

__model_hash = {
        "DQN_1": Source.Models.DQN_Model_1,
        "DQN_2": Source.Models.DQN_Model_2,
        "DQN_3": Source.Models.DQN_Model_3
        }

def ModelSelect(model_name):
    return __model_hash[model_name]


# ============================
# NOISE
# - Ornstein-Uhlenbeck process
# ============================
class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

# ================================
# TRACKER
# - Unified metric tracking system
# ================================
class Tracker:
    def __init__(self, window = 100):
        self.rewards = deque(maxlen=window)
        self.levels = []
        self.window = window
        
    def mean(self):
        return np.mean(self.rewards), np.mean(self.levels[-self.window:])
    
    def maximum(self):
        return np.max(self.rewards), np.max(self.levels[-self.window:])
    
    def save_levels(self, folder):
        if not os.path.exists("./Saved Models"):
            os.mkdir("./Saved Models")
            
        if not os.path.exists("./Saved Models/{}".format(folder)):
            os.mkdir("./Saved Models/{}".format(folder))
        np.savetxt("./Saved Models/{}/level_scores.txt".format(folder), np.array(self.levels), delimiter=",")
        
    def load_levels(self, folder):
        if os.path.exists("./Saved Models/{}/level_scores.txt".format(folder)):
            self.levels = np.loadtxt("./Saved Models/{}/level_scores.txt".format(folder), delimiter=",").tolist()
            self.rewards.clear()
            
    def add(self, reward, level):
        self.rewards.append(reward)
        self.levels.append(level)
        
    def display(self, epoch, total_epochs, clock, end="\n"):
        print("[{}/{} | {:0.2f}s] Mean: {:0.4f} | Max: {:0.4f} | Mean Lvl: {:0.4f} | Max Lvl: {}. {}".format(
                epoch+1, total_epochs, clock,
                np.mean(self.rewards), np.max(self.rewards), 
                np.mean(self.levels[-100:]), np.max(self.levels[-100:]), " "*20),
             end=end)

# =========================================
# CONVERTER
# - transforms inputs to accionable formats
# =========================================
class Converter:
    oh6 = {
        '100000': [0,0,0,0],
        '010000': [1,0,0,0],
        '001000': [2,0,0,0],
        '000100': [1,0,1,0],
        '000010': [0,1,0,2],
        '000001': [0,2,0,1]
        }
    
    @staticmethod
    def ProcessState(state):
        return np.rollaxis(np.array([state]), 3, 1)

    @staticmethod
    def Action2OneHot(action):
        index  = action[0]*18 + action[1]*6 + action[2]*3 + action[3]
        output = [1 if x == index else 0 for x in range(54)]
        return output

    @staticmethod
    def OneHot2Action(onehot):
        value = np.argmax(onehot)
        return [int(value/18)%3, int(value/6)%3, int(value/3)%2, value%3]
    
    def OneHot2Action6(onehot):
        return Converter.oh6["".join(str(i) for i in onehot)]
        