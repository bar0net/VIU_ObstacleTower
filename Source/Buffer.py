# -*- coding: utf-8 -*-
"""
@author: Jordi Tudela Alcacer
"""

from collections import deque, namedtuple
import numpy as np
import random

class Buffer:
    def __init__(self, buffer_size, batch_size, seed = 0):
        self.batch_size = batch_size
        self.memory = deque(maxlen = int(buffer_size))
        self.experience = namedtuple("experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def __len__(self):
        return len(self.memory)
    
    @staticmethod
    def Action2OneShoot(action):
        curr_action = [0]*11
        curr_action[action[0]] = 1
        curr_action[action[1]+3] = 1
        curr_action[action[2]+6] = 1
        curr_action[action[3]+8] = 1
        return curr_action
    
    def add(self, state, action, reward, next_state, done):
        item = self.experience(state, action, reward, next_state, float(done))
        self.memory.append(item)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states =      np.vstack([e.state        for e in experiences if e is not None])
        #actions =     np.vstack([Buffer.Action2OneShoot(e.action) for e in experiences if e is not None])
        actions =     np.vstack([e.action       for e in experiences if e is not None])
        rewards =     np.vstack([e.reward       for e in experiences if e is not None])
        next_states = np.vstack([e.next_state   for e in experiences if e is not None])
        dones =       np.vstack([e.done         for e in experiences if e is not None])

        return (states, actions, rewards, next_states, dones)
    
    def active(self):
        return len(self.memory) >= self.batch_size
        