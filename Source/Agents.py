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
        
        error = (Q_expected - Q_targets).pow(2)
        loss = error.mean()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        Agent.soft_update(self.local, self.target, self.tau)
        
        self.buffer.learn_update(values=error.squeeze().cpu().data.numpy(), debug=True)
        
        
        
class DistDDQN_Agent(DoubleDQN_Agent):
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
            value = (action_values * self.local.support).sum(2)
            output = value.max(1)[1].item()
        self.local.train()
        
        return [output.cpu().numpy()]
# ==========================================
#                  PPO
# ==========================================
class PPO_Agent(Agent):
    def __init__(self, state_size, action_size, model_name, gradient_steps = 16, batch_size = 64, 
                 initial_discount = 0.995, epsilon=0.1,  learning_rate = 1e-4, n_iter = 1, seed = 0, device = 'cpu'):
        
        super().__init__(state_size, action_size, "PPO", None, n_iter, seed, device)
        
        self.steps = gradient_steps
        self.discount = initial_discount
        self.batch_size = batch_size
        self.epsilon = epsilon
        
        self.model = ModelSelect(model_name)(state_size, action_size, seed=seed, device=device).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def step_end(self, probs, states, actions,  returns, advantages, train=True): 
        if train:
            for _ in range(self.steps):
                mini_probs, mini_states, mini_actions, mini_returns, mini_advantages = self._get_minibatch(probs, states, actions, returns, advantages)        
                target = self._clip(mini_probs, mini_states, mini_actions, mini_returns, mini_advantages)
                
                self.optimizer.zero_grad()
                target.backward()
                self.optimizer.step()
                del target
        
    def act(self, state, train=True):
        torch_state = torch.from_numpy(state).float().to(self.device)
        return self.model(torch_state)
    
    def save(self, prefix="", suffix=""):
        if not os.path.exists("./Saved Models"):
            os.mkdir("./Saved Models")
            
        folder = self.get_folder(prefix=prefix, suffix=suffix)
            
        if not os.path.exists("./Saved Models/{}".format(folder)):
            os.mkdir("./Saved Models/{}".format(folder))
            
        torch.save(self.model.state_dict(), "./Saved Models/{}/model.pth".format(folder))
        
    def load_weights(self, prefix="", suffix=""):
        folder = self.get_folder(prefix=prefix, suffix=suffix)
        if os.path.exists("./Saved Models/{}/level_scores.txt".format(folder)):
            print("Reloading existing model")
            self.model.load_state_dict(torch.load("./Saved Models/{}/model.pth".format(folder)))
            
    def _learn(self):
        return
            
    def _get_minibatch(self, probs, states, actions, rewards, values):        
        idx = np.random.randint(0, states.shape[0], self.batch_size)
        return probs[idx,:], states[idx,:], actions[idx,:], rewards[idx,:], values[idx,:]
       
    def _clip(self, probs, states, actions, returns, advantages):       
        actions     = torch.tensor(actions, dtype=torch.float, device=self.device)
        returns     = torch.tensor(returns, dtype=torch.float, device=self.device)
        probs       = torch.tensor(probs, dtype=torch.float, device=self.device)
        states      = torch.tensor(states, dtype=torch.float, device=self.device)
        advantages  = torch.tensor(advantages, dtype=torch.float, device=self.device)
        
        dist, value = self.model(states)
        entropy = dist.entropy().mean()
        new_probs = dist.log_prob(actions)
        
        ratio = (new_probs - probs).exp()
        
        old_rewards = advantages * ratio
        clipped_ratio   = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        
        actor_loss  = torch.min(old_rewards, clipped_ratio).mean()
        critic_loss =  (returns-value).pow(2).mean()
        
        return actor_loss + 0.5 * critic_loss - 0.001 * entropy
    
    @staticmethod
    def normalize_rewards(next_value, values, rewards, masks, gamma=0.99, tau=0.95):
        values = np.vstack( (values, next_value) )
        x, returns = 0, []
        N = len(rewards)
        
        for i in reversed(range(N)):
            if (rewards[i] > 0):
                x = rewards[i] - values[i]
            else:
                delta = rewards[i] + gamma * values[i+1] * masks[i] - values[i]
                x = delta + gamma * tau * masks[i] * x
            returns.insert(0, x + values[i])
        return returns
        
    
    
    
    
    
    
    
    
    
    
    
    