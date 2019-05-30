import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
import math

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        #self.layer0 = nn.Linear(6, 6)
        self.layer1 = nn.Linear(6, 2)
        self.layer2 = nn.Linear(2, 6)
        #self.layer3 = nn.Linear(6, 6)
    
    def encode(self, x):
        #x = F.leaky_relu(self.layer0(x))
        #x = F.leaky_relu(self.layer1(x))
        #x = self.layer0(x)
        x = self.layer1(x)
        return x

    def decode(self, x):
        #x = F.leaky_relu(self.layer2(x))
        x = self.layer2(x)
        #x = self.layer3(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

