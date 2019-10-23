# -*- coding: utf-8 -*-
"""
@author: Jordi Tudela Alcacer
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    