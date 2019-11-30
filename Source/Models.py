# -*- coding: utf-8 -*-
"""
@author: Jordi Tudela Alcacer
"""
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

# =====================================================
# Noisy linear layer with independent Gaussian noise
# Code by: Kai Arulkumaran  (https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py#L10-L37)
class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.sigma_init = sigma_init
    self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = nn.Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      nn.init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      nn.init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      nn.init.constant(self.sigma_weight, self.sigma_init)
      nn.init.constant(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)
# =============================================================
    
    

class DQN_Model_1(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, seed = 0, device='cpu'):
        super(DQN_Model_1,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
                
        # Model Layers
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=9, stride=4), # 168x168 -> 40x40
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2), # 40x40 -> 18x18
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2), # 18x18 -> 8x8
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1), # 8x8 -> 6x6
                nn.ReLU(),
                Flatten(),
                nn.Linear(32*6*6, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
                )
        
    def forward(self, state):
        return self.model(state)



class DQN_Model_2(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, seed = 0, device='cpu'):
        super(DQN_Model_2,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
                
        # Model Layers
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=9, stride=4), # 168x168 -> 40x40
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2), # 40x40 -> 18x18
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), # 18x18 -> 8x8
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2), # 8x8 -> 3x3
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1), # 3x3 -> 1x1
                nn.ReLU(),
                Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, action_size),
                nn.ReLU(),
                nn.Linear(action_size, action_size)
                )
        
    def forward(self, state):
        return self.model(state)
    

class DQN_Model_3(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, seed = 0, device='cpu'):
        super(DQN_Model_3,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
                
        # Model Layers
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1), # 168x168 -> 166x166
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2), # 166x166 -> 82x82
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2), # 82x82 -> 40x40
                nn.ReLU(),
                
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1), # 40x40 -> 38x38
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2), # 38x38 -> 18x18
                nn.ReLU(),
                
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2), # 18x18 -> 8x8
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), # 8x8 -> 6x6
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), # 6x6 -> 2x2
                nn.ReLU(),
                
                Flatten(),
                nn.Linear(128*2*2, 128),
                nn.ReLU(),
                nn.Linear(128, action_size),
                nn.ReLU(),
                nn.Linear(action_size, action_size)
                )
        
    def forward(self, state):
        return self.model(state)  
    


class DDQN_Model_1(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, seed = 0, device='cpu'):
        super(DDQN_Model_1,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
        
        conv_out = [(state_size[0] - 8) / 4, (state_size[1] - 8) / 4]
        conv_out = [(conv_out[0]   - 4) / 2, (conv_out[1]   - 4) / 2]
        conv_out = [(conv_out[0]   - 2) / 2, (conv_out[1]   - 2) / 2]
        conv_out = [(conv_out[0]   - 2),     (conv_out[1]   - 2)]
        
        # Model Layers
        self.conv_model = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=9, stride=4), # 168x168 -> 40x40
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2), # 40x40 -> 18x18
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2), # 18x18 -> 8x8
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1), # 8x8 -> 6x6
                nn.ReLU(),
                Flatten()
                )
        
        self.state_head = nn.Sequential(
                nn.Linear(32*conv_out[0]*conv_out[1], 128),
                nn.ReLU(),
                nn.Linear(128,action_size),
                nn.ReLU()
                )
        
        self.action_head = nn.Sequential(
                nn.Linear(32*conv_out[0]*conv_out[1], 128),
                nn.ReLU(),
                nn.Linear(128,action_size),
                nn.ReLU()
                )
        
        self.out_layer = nn.Linear(action_size, action_size)
        
    def forward(self, state):
        
        x = self.conv_model(state)
        v = self.state_head(x)
        a = self.action_head(x)

        x = v + (a - torch.mean(a, dim=0))
                
        return self.out_layer(x)
    

class Noisy_DDQN_Model_1(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, sigma=0.017, seed = 0, device='cpu'):
        super(Noisy_DDQN_Model_1,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
        
        conv_out = [(state_size[0] - 8) / 4, (state_size[1] - 8) / 4]
        conv_out = [(conv_out[0]   - 4) / 2, (conv_out[1]   - 4) / 2]
        conv_out = [(conv_out[0]   - 2) / 2, (conv_out[1]   - 2) / 2]
        conv_out = [(conv_out[0]   - 2),     (conv_out[1]   - 2)]
        
        # Model Layers
        self.conv_model = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=9, stride=4), # 168x168 -> 40x40
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2), # 40x40 -> 18x18
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2), # 18x18 -> 8x8
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1), # 8x8 -> 6x6
                nn.ReLU(),
                Flatten()
                )
        
        self.state_head = nn.Sequential(
                nn.Linear(32*conv_out[0]*conv_out[1], 128),
                nn.ReLU(),
                nn.Linear(128,action_size),
                nn.ReLU()
                )
        
        self.action_head = nn.Sequential(
                nn.Linear(32*conv_out[0]*conv_out[1], 128),
                nn.ReLU(),
                nn.Linear(128,action_size),
                nn.ReLU()
                )
        
        self.out_layer = NoisyLinear(in_features=action_size, out_features=action_size, sigma_init=sigma)
        
    def forward(self, state):
        
        x = self.conv_model(state)
        v = self.state_head(x)
        a = self.action_head(x)

        x = v + (a - torch.mean(a, dim=0))
                
        return self.out_layer(x)
    
    
class Distributional_DDQN_Model_1(nn.Module):
    def __init__(self, state_size, action_size, atoms= 8, v_min=0, v_max=1, input_channels=3, sigma=0.017, seed = 0, device='cpu'):
        super(Distributional_DDQN_Model_1,self).__init__()
        
        self.action_size = action_size
        self.atoms = atoms
        
        self.dqn = Noisy_DDQN_Model_1(state_size, atoms*action_size, input_channels, sigma, seed, device)
        support = torch.linspace(v_min, v_max, atoms)
        self.register_buffer('support', support)
        
    def forward(self, state):
        
        x = self.dqn(state)
        
        return x.view(-1,self.action_size, self.atoms)
    
class Actor_Critic_Model(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, std=0.0, seed = 0, device='cpu'):
        super(Actor_Critic_Model,self).__init__()
        
        self.critic = DQN_Model_1(state_size, 1, input_channels, seed, device)
        self.actor  = DDQN_Model_1(state_size, action_size, input_channels, seed, device)
        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)
        
    def forward(self, state):
        value = self.critic(state)

        mu  = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
                
        return Normal(mu,std), value
    
class PPO_Model_1(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, seed = 0, device='cpu'):
        super(PPO_Model_1,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
                
        # Model Layers
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=9, stride=4), # 168x168 -> 40x40
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2), # 40x40 -> 18x18
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2), # 18x18 -> 8x8
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1), # 8x8 -> 6x6
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), # 6x6 -> 2x2
                nn.ReLU(),
                Flatten(),
                nn.Linear(64*2*2, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
                )
        
    def forward(self, state):
        return self.model(state)
    
class PPO_Model_2(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, seed = 0, device='cpu'):
        super(PPO_Model_2,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
                
        # Model Layers
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=9, stride=4), # 168x168 -> 40x40
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2), # 40x40 -> 18x18
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2), # 18x18 -> 8x8
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1), # 8x8 -> 6x6
                nn.ReLU(),
                Flatten(),
                nn.Linear(32*6*6, 512),
                nn.ReLU(),
                nn.Linear(512, action_size)
                )
        
    def forward(self, state):
        return self.model(state)
    
    
    
class PPO_Model_3(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, seed = 0, device='cpu'):
        super(PPO_Model_3,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
                
        # Model Layers
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=9, stride=4), # 168x168 -> 40x40
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2), # 40x40 -> 18x18
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2), # 18x18 -> 8x8
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1), # 8x8 -> 6x6
                nn.ReLU(),
                Flatten(),
                nn.Linear(32*6*6, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
                )
        
    def forward(self, state):
        return self.model(state)
    
class PPO_Model_4(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, seed = 0, device='cpu'):
        super(PPO_Model_4,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
                
        # Model Layers        
        self.cnn1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=9, stride=4) # 168x168 -> 40x40
        
        
        self.cnn2 =  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2) # 40x40 -> 18x18
        self.cnn2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1) # 40x40 -> 38x38
        self.cnn2c = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2) # 38x38 -> 18x18
        
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2) # 18x18 -> 8x8
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1) # 8x8 -> 6x6
        self.cnn5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1) # 6x6 -> 4x4
        self.cnn4b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1) # 6x6 -> 4x4
        
        self.flatten = Flatten()
        self.fcn1 = nn.Linear(32*4*4, 128)
        self.fcn2 = nn.Linear(128, action_size)
        
    def forward(self, state):
        x = F.relu(self.cnn1(state))
        #print("A", x.shape)
        x1 = F.relu(self.cnn2(x))
        x2 = F.relu(self.cnn2b(x))
        x2 = F.relu(self.cnn2c(x2))
        
        #print("B", x1.shape, x2.shape)
        x = F.relu(x2-x1)
        
        #print("C", x.shape)
        x = F.relu(self.cnn3(x))
        
        #print("D", x.shape)
        x1 = F.relu(self.cnn4b(x))
        x2 = F.relu(self.cnn4(x))
        x2 = F.relu(self.cnn5(x2))

        #print("E", x1.shape, x2.shape)
        x = F.relu(x2 - x1)
    
        #print("F", x.shape)
        x = self.flatten(x)

        #print("G", x.shape)
        x = F.relu(self.fcn1(x))
        
        #print("H", x.shape)
        return self.fcn2(x)
    
    
class PPO_Model_5(nn.Module):
    def __init__(self, state_size, action_size, input_channels=3, seed = 0, device='cpu'):
        super(PPO_Model_5,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
                
        # Model Layers        
        self.cnn1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=9, stride=4) # 168x168 -> 40x40
        
        
        self.cnn2 =  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2) # 40x40 -> 18x18
        self.cnn2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1) # 40x40 -> 38x38
        self.cnn2c = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2) # 38x38 -> 18x18
        
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2) # 18x18 -> 8x8
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1) # 8x8 -> 6x6
        self.cnn5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1) # 6x6 -> 4x4
        self.cnn4b = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1) # 6x6 -> 4x4
        
        self.flatten = Flatten()
        self.fcn1 = nn.Linear(32*4*4, 128)
        self.fcn2 = nn.Linear(128, action_size)
        
    def forward(self, state):
        x = F.relu(self.cnn1(state))
        #print("A", x.shape)
        x1 = F.relu(self.cnn2(x))
        x2 = F.relu(self.cnn2b(x))
        x2 = F.relu(self.cnn2c(x2))
        
        #print("B", x1.shape, x2.shape)
        x = torch.cat((x1,x2), dim=1)
        
        #print("C", x.shape)
        x = F.relu(self.cnn3(x))
        
        #print("D", x.shape)
        x1 = F.relu(self.cnn4b(x))
        x2 = F.relu(self.cnn4(x))
        x2 = F.relu(self.cnn5(x2))

        #print("E", x1.shape, x2.shape)
        x = torch.cat((x1,x2), dim=1)
    
        #print("F", x.shape)
        x = self.flatten(x)

        #print("G", x.shape)
        x = F.relu(self.fcn1(x))
        
        #print("H", x.shape)
        return self.fcn2(x)
    
    
    