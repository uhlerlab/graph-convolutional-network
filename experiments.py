import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from IPython import embed
import random
import math
from gcn import GCN, Graph, train
import matplotlib.pyplot as plt

w = torch.FloatTensor([[0.5,0.],[-1.,0.],[.75,-0.5],[.75,1.],[0.,0.5],[0,-1.5]])

num_samples = 50

x = [torch.FloatTensor([[1 + random.random()], [-1 - random.random()]]) for i in range(num_samples)]

# x = [torch.FloatTensor([[((-1)**(i % 2))*random.random()], [((-1)**(i % 2))*random.random()]]) for i in range(num_samples)]

epsilon = 0.0

y = [torch.mm(w,x_i).view(-1) + epsilon*torch.randn(6) for x_i in x]

edge_list = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]

net = GCN(Graph(6,edge_list))

train(y, net)

latent = [net.encode(i) for i in y]

plt.figure(1)
plt.subplot(211)
plt.scatter([i[0].item() for i in x], [i[1].item() for i in x])

plt.subplot(212)
plt.scatter([i[0].item() for i in latent], [i[1].item() for i in latent])

plt.show()

embed()