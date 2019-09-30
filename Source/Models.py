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

class DQN_Model(nn.Module):
    def __init__(self, state_size, action_size, seed = 0, device='cpu'):
        super(DQN_Model,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_dim = [max(action_size) - x for x in action_size]
        self.device = device
        
        # Model Layers
        self.conv_l1 = nn.Conv2d( 3, 16, (3,3)).to(device)                  # 168x168 -> 166x166
        self.conv_l2 = nn.Conv2d(16, 16, (3,3)).to(device)                  # 166x166 -> 164x164
        self.conv_l3 = nn.Conv2d(16, 32, (5,5), stride=(2,2)).to(device)    # 164x164 -> 80x80
        
        self.res1_l1 = nn.Conv2d(32, 16, (1,1)).to(device)
        self.res1_l2 = nn.Conv2d(16, 16, (3,3), padding=(1,1)).to(device)
        self.res1_l3 = nn.Conv2d(16, 32, (1,1)).to(device)
        
        self.res2_l1 = nn.Conv2d(32, 16, (1,1)).to(device)
        self.res2_l2 = nn.Conv2d(16, 16, (3,3), padding=(1,1)).to(device)
        self.res2_l3 = nn.Conv2d(16, 32, (1,1)).to(device)
        
        self.conv_l4 = nn.Conv2d(32, 64, (5,5), stride=(2,2)).to(device)    # 80x80 -> 38x38
        
        self.res3_l1 = nn.Conv2d(64, 32, (1,1)).to(device)
        self.res3_l2 = nn.Conv2d(32, 32, (3,3), padding=(1,1)).to(device)
        self.res3_l3 = nn.Conv2d(32, 64, (1,1)).to(device)
        
        self.res4_l1 = nn.Conv2d(64, 32, (1,1)).to(device)
        self.res4_l2 = nn.Conv2d(32, 32, (3,3), padding=(1,1)).to(device)
        self.res4_l3 = nn.Conv2d(32, 64, (1,1)).to(device)
        
        self.conv_l5 = nn.Conv2d(64,  64, (3,3), stride=(2,2)).to(device)    # 38x38 -> 18x18
        self.conv_l6 = nn.Conv2d(64, 128, (3,3), stride=(2,2)).to(device)    # 18x18 ->  8x8
        self.conv_l7 = nn.Conv2d(128,128, (3,3)).to(device)                  #  8x8  ->  6x6
        self.conv_l8 = nn.Conv2d(128,256, (3,3)).to(device)                  #  6x6  ->  4x4
        self.conv_l9 = nn.Conv2d(256,256, (3,3)).to(device)                  #  4x4  ->  2x2
        
        self.fc1 =  nn.Linear(1024,128).to(device)
        
        self.out0 = nn.Linear(128,action_size[0]).to(device)
        self.out1 = nn.Linear(128,action_size[1]).to(device)
        self.out2 = nn.Linear(128,action_size[2]).to(device)
        self.out3 = nn.Linear(128,action_size[3]).to(device)
        
        #for key, layer in self.layers.items():
            #layer.weight.data.uniform_(*hidden_init(layer))
        
    def forward(self, state):
        x = self.conv_l1(state)
        x = F.relu(self.conv_l2(x))
        x = F.relu(self.conv_l3(x))
        
        r = F.relu(self.res1_l1(x))
        r = F.relu(self.res1_l2(r))
        r = self.res1_l3(r)
        x = r + x
        
        r = F.relu(self.res2_l1(x))
        r = F.relu(self.res2_l2(r))
        r = self.res2_l3(r)
        x = r + x
        
        x = F.relu(self.conv_l4(x))
        
        r = F.relu(self.res3_l1(x))
        r = F.relu(self.res3_l2(r))
        r = self.res3_l3(r)
        x = r + x
        
        r = F.relu(self.res4_l1(x))
        r = F.relu(self.res4_l2(r))
        r = self.res4_l3(r)
        x = r + x
        
        x = F.relu(self.conv_l5(x)) 
        x = F.relu(self.conv_l6(x)) 
        x = F.relu(self.conv_l7(x)) 
        x = F.relu(self.conv_l8(x)) 
        x = F.relu(self.conv_l9(x))  
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        
        
        out0 = F.softmax(self.out0(x), dim=1)
        out1 = F.softmax(self.out1(x), dim=1)
        out2 = F.softmax(self.out2(x), dim=1)
        out3 = F.softmax(self.out3(x), dim=1)
        
        return torch.cat( (out0, out1, out2, out3), 1 ).to(self.device)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    