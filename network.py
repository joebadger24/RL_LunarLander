import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

    
class Network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(in_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        
        return output
    

    # These functions allow saving and loading of models
    def save(self, filename='network/latest.pt'):
        if not os.path.exists("network"):
            os.makedirs("network")
        torch.save(self.state_dict(), filename)

    def load(self, start_new, filename='network/latest.pt'):
        if start_new == False:
            try:
                self.load_state_dict(torch.load(filename))
                print("Loaded latest file!")
            except:
                print("No file found!")
        else:
            print("Starting training from scratch")

        
            

