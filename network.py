from torch import nn
import torch

class NN(nn.Module):
    def __init__(self, resnet_pretrained):
        super().__init__()
        self.resnet_pretrained = resnet_pretrained
        self.fc1 = nn.Linear(1000, 4)
        
        
    def forward(self, x):
        x = torch.relu(self.resnet_pretrained(x))
        x = self.fc1(x)

        return x