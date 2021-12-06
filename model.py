import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_1 = nn.Linear(self.input_size, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, output_size)
        
    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        z1 = self.layer_1(x)
        z2 = self.layer_2(F.relu(z1))
        output = self.output_layer(z2)
        return output