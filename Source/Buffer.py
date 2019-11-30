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
    
    def update(self):
        pass
        
    def learn_update(self, values = None, debug = True):
        pass
    
class SortedBuffer(Buffer):
    def __init__(self, buffer_size, batch_size, seed = 0):
        self.batch_size = batch_size
        self.memory = []
        self.experience = namedtuple("experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        
    def add(self, state, action, reward, next_state, done):
        item = self.experience(state, action, reward, next_state, float(done))
        self.memory.append(item)
        
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)
            
    def update(self):
        mean = sum( map(lambda x: x.reward, self.memory) )
        
        # We want lower std dev values at the front so we can
        # pop front and append to back (it's convenient!)
        self.memory = sorted(self.memory, reverse=True, key=lambda x: abs(mean-x.reward) + 0.00001 * random.random())
        
        
class PriorityBuffer(Buffer):
    def __init__(self, buffer_size, batch_size, default_priority = 1, seed = 0):
        self.batch_size = batch_size
        self.memory = deque(maxlen = int(buffer_size))
        self.experience = namedtuple("experience", field_names = ["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.default_p = default_priority
        self.indices = None
        
    def add(self, state, action, reward, next_state, done):
        item = self.experience(state, action, reward, next_state, float(done), self.default_p)
        self.memory.append(item)
        
    def sample(self):
        sum_of_prios = sum([x.priority for x in self.memory])
        weights = [x.priority / sum_of_prios for x in self.memory]
        
        N = len(self.memory)
        self.indices = np.random.choice(range(N), size=self.batch_size, p=weights)
        self.weights = [weights[i] for i in self.indices]
        
        experiences = [self.memory[i] for i in self.indices]

        states =      np.vstack([e.state        for e in experiences if e is not None])
        actions =     np.vstack([e.action       for e in experiences if e is not None])
        rewards =     np.vstack([e.reward       for e in experiences if e is not None])
        next_states = np.vstack([e.next_state   for e in experiences if e is not None])
        dones =       np.vstack([e.done         for e in experiences if e is not None])

        return (states, actions, rewards, next_states, dones)
    
    def learn_update(self, values = None, debug = True):
        if debug:   
            if type(self.indices) == type(None):
                ValueError("Indices not set")
            
            if type(values) != list: 
                TypeError("Invalid type of values")
                
            if len(values) != len(self.indices):
                ValueError("Invalid values. Expected a list of size {}, got a vector of size {}".format(len(self.indices), len(values)))
                
        for i, idx in enumerate(self.indices):
            
            w   = self.weights[i]
            p   = values[i]
            
            self.memory[idx]._replace(priority=(1e-5 + p*w))
        
        
        
        
        