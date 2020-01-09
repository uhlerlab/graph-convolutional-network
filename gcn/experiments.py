import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
import random
import math
import numpy
from gcn import GCN, Graph, train
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from fully_connected import AE

print(torch.cuda.is_available())

w = torch.FloatTensor([[0.5,0.],[-1.,0.],[.75,0.],[0.,1.],[0.,0.5],[0,-1.5]])

num_samples = 1000

x = [torch.FloatTensor([[1 + random.random()], [-1 - random.random()]]) for i in range(num_samples)]

#x = [torch.FloatTensor([[((-1)**(i % 2))*random.random()], [((-1)**(i % 2))*random.random()]]) for i in range(num_samples)]

epsilon = 0.0

y = [torch.mm(w,x_i).view(-1) + epsilon*torch.randn(6) for x_i in x]

edge_list = [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)]

net = GCN(Graph(6,edge_list))
AE = AE()
if torch.cuda.is_available():
     net.cuda()
     AE.cuda()
loss_gcn = train(y, net)
loss_fcn = train(y, AE)
fig = plt.figure(1)
plt.title("Loss")
plt.plot(loss_gcn, "b")
plt.plot(loss_fcn, "r")
plt.savefig("losses.png")
latent = [net.encode(i.cuda()) for i in y]
AElatent = [AE.encode(i.cuda()) for i in y]

latent0 = [i.data[0].item() for i in latent]
latent1 = [i.data[1].item() for i in latent]
AElatent0 = [i.data[0].item() for i in AElatent]
AElatent1 = [i.data[1].item() for i in AElatent]
print("gcn correlation = ",numpy.corrcoef(latent0,latent1)[0][1])
print("fcn correlation = ", numpy.corrcoef(AElatent0, AElatent1)[0][1])
xmin = -2
xmax = 2
ymin = -2
ymax = 2

fig = plt.figure(2)
ax1 = fig.add_subplot(211)
#axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#axes.set_ylim([ymin,ymax])
ax1.scatter(latent0, latent1)
ax1.set_title("Graph Convolutional Network")

ax2 = fig.add_subplot(212)
#axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#axes.set_ylim([ymin,ymax])
ax2.scatter(AElatent0, AElatent1)
ax2.set_title("Fully Connected Network")
plt.savefig('plot.png')

with open('results', 'w') as f:
        f.write(str(numpy.corrcoef(latent0, latent1)[0][1])+ '\n')
        f.write(str(numpy.corrcoef(AElatent0, AElatent1)[0][1]))
