# -*- coding: utf-8 -*-
"""
@author: Jordi Tudela Alcacer
"""
import os
import random
import datetime
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim

from Source.Utils  import ModelSelect
from Source.Buffer import Buffer

# ============================================================================
# BASE AGENT CLASS
# ============================================================================
class Agent(ABC):
    def __init__(self, state_size, action_size, agent_type, buffer = None, n_iter = 1, seed = 0, device = 'cpu'):
        if not isinstance(buffer, Buffer):
            TypeError("Incorrect type for buffer")
        
        if int(n_iter) <= 0:
            ValueError("n_iter must be greater than 1")
        
        self.state_size  = state_size
        self.action_size = action_size
        self.buffer = buffer
        self.n_iter = int(n_iter)
        self.seed = random.seed(seed)
        self.device = device
        
        self.type = agent_type
        now = datetime.datetime.now()
        self.date = "{}{}{}".format(now.year,now.month,now.day)
    
    def step_begin(self):
        pass
    
    def step_update(self):
        if self.buffer != None and self.buffer.active():
            for _ in range(self.n_iter):
                self._learn()
    
    def step_end(self):
        pass
    
    @abstractmethod
    def act(self, state, train=True):
        pass
    
    def reset(self):
        pass
    
    @abstractmethod
    def save(self, prefix="", suffix=""):
        pass
    
    @abstractmethod
    def load_weights(self, prefix="", suffix=""):
        pass
    
    def get_folder(self, prefix="", suffix=""):
        folder = self.type + "-" + self.date
        if prefix != "":
            folder = prefix + "_" + folder
        if suffix != "":
            folder += "_" + suffix
        return folder
    
    @abstractmethod
    def _learn(self):
        pass
    
    def _param_update(self):
        pass
    
    @staticmethod    
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model:  PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float):  interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            

# ==========================================
#                  Random
# ==========================================          
class Random_Agent(Agent):
    def __init__(self, state_size, action_size, seed = 0, device = 'cpu'):
        super().__init__(state_size, action_size, "Random", None, 0, seed, device)
        
    def act(self, state, train=True):
        output = [0]*self.action_size
        output[random.randint(0,self.action_size-1)] = 1
        return [output]  
    
    def _learn(self):
        pass
    
    def save(self, prefix="", suffix=""):
        pass
    
    def load_weights(self, prefix="", suffix=""):
        pass
    
        
class SingleAction_Agent(Agent):
    def __init__(self, state_size, action_size, action, seed = 0, device = 'cpu'):
        super().__init__(state_size, action_size, "Random", None, 0, seed, device)
        self.action = action
        
    def act(self, state, train=True):
        return self.action
    
    def _learn(self):
        pass
    
    def save(self, prefix="", suffix=""):
        pass
    
    def load_weights(self, prefix="", suffix=""):
        pass
    
# ==========================================
#                  DQN
# ==========================================
            
class DoubleDQN_Agent(Agent):
    def __init__(self, state_size, action_size, buffer, model_name="DQN_1", gamma = 0.99, tau = 1e-3, epsilon_start = 1.0,
                 epsilon_end = 1e-3, epsilon_steps = 1000, learning_rate = 1e-3, 
                 n_iter = 1, seed = 0, device = 'cpu'):
        super().__init__(state_size, action_size, "DoubleDQN", buffer, n_iter, seed, device)
        
        self.gamma = gamma
        self.tau   = tau
        self.epsilon = defaultdict(lambda:epsilon_start)
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_steps
        self.epsilon_end = epsilon_end
        
        self.local  = ModelSelect(model_name)(state_size, action_size, seed=seed, device=device).to(device)
        self.target = ModelSelect(model_name)(state_size, action_size, seed=seed, device=device).to(device)
        self.optim  = optim.Adam(self.local.parameters(), lr=learning_rate)
        self.current_level = 0
        
    def step_begin(self):
        self.current_level = 0
        
    def step_end(self):
        for key, value in self.epsilon.items():
            if key > self.current_level:
                break
            self.epsilon[key] = max(self.epsilon[key]-self.epsilon_decay, self.epsilon_end)
    
    def act(self, state, train=True):
        # Epsilon-greedy Policy
        output = [0]*self.action_size
        if train and random.random() < self.epsilon[self.current_level]:
            output[random.randint(0,self.action_size-1)] = 1
            return [output]
        
        torch_state = torch.from_numpy(state).float().to(self.device)
        self.local.eval()
        with torch.no_grad():
            action_values = self.local(torch_state)
        self.local.train()
        
        action = np.argmax(action_values.cpu().numpy())
        output[action] = 1
        return [output]
    
    def save(self, prefix="", suffix=""):
        if not os.path.exists("./Saved Models"):
            os.mkdir("./Saved Models")
            
        folder = self.get_folder(prefix=prefix, suffix=suffix)
            
        if not os.path.exists("./Saved Models/{}".format(folder)):
            os.mkdir("./Saved Models/{}".format(folder))
            
        torch.save(self.local.state_dict(), "./Saved Models/{}/local.pth".format(folder))
        torch.save(self.target.state_dict(), "./Saved Models/{}/target.pth".format(folder))
        
    def load_weights(self, prefix="", suffix=""):
        folder = self.get_folder(prefix=prefix, suffix=suffix)
        if os.path.exists("./Saved Models/{}/level_scores.txt".format(folder)):
            print("Reloading existing model")
            self.local.load_state_dict(torch.load("./Saved Models/{}/local.pth".format(folder)))
            self.target.load_state_dict(torch.load("./Saved Models/{}/local.pth".format(folder)))
    
    def _learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        states      = torch.from_numpy(states).float().to(self.device)
        actions     = torch.from_numpy(actions).long().to(self.device)
        rewards     = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones       = torch.from_numpy(dones).float().to(self.device)
        
        Q_targets_next = self.target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets  = rewards + (self.gamma * Q_targets_next * (1-dones))
        Q_expected = self.local(states).gather(1,actions)[:,0].unsqueeze(1)
        
        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        Agent.soft_update(self.local, self.target, self.tau)
            