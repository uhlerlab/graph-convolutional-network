import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributions as dist
from torch.autograd import Variable
import torch.optim as optim
import random
import math
import numpy
from numpy import linalg as LA
from gcn import GraphLinearLayer, GraphPool, Graph, train
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import time

def get_precision_matrix(edge_list, values, dim):
    matrix = [[0 for i in range(dim)] for i in range(dim)]
    for i in range(len(edge_list)):
        j,k = edge_list[i]
        matrix[j][k] = values[i]
        matrix[k][j] = values[i]
    for i in range(dim):
        matrix[i][i] = 1.0
    return torch.FloatTensor(matrix)

edge_list = [(0,1), (0,2), (0,3), (2,4), (2,5), (4,5), (3,6), (3,7), (6,7)]
omega = [get_precision_matrix(edge_list, [0.4, 0.4, 0.4, -0.4, -0.4, -0.4, 0.4, 0.4, 0.4], 8), 
	get_precision_matrix(edge_list, [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, -0.4, -0.4, -0.4], 8)]

true_dist = [dist.multivariate_normal.MultivariateNormal(torch.zeros(8), precision_matrix = omega[0]),
	 dist.multivariate_normal.MultivariateNormal(0.1*torch.ones(8), precision_matrix = omega[0])]
data = [true_dist[i%2].sample() for i in range(2)]
print(data)
print(data[0].numpy()/data[1].numpy())
test_data = [true_dist[i%2].sample() for i in range(1000)]
test_labels = [i%2 for i in range(1000)]
labels = [i%2 for i in range(2)]
y = torch.FloatTensor([labels])

class GCN(nn.Module):

    def __init__(self, input_graph):
        super(GCN, self).__init__()

        input_graph = input_graph

        self.mask0 = nn.Parameter(input_graph.mask, requires_grad = False)
        
        self.layer0 = GraphLinearLayer(input_graph.vertices, self.mask0)
        self.layer1 = GraphLinearLayer(input_graph.vertices, self.mask0)
        #self.layer2 = GraphLinearLayer(input_graph.vertices, mask0)
        #self.layer3 = GraphLinearLayer(input_graph.vertices, mask0)
        
        self.avg = nn.Parameter(torch.FloatTensor([[1.0/3, 1.0/3, 1.0/3]]), requires_grad = False) 
        contraction0 = [[0,1],[2,4,5],[3,6,7]]
        self.pool = GraphPool(input_graph.vertices, contraction0)
        graph1 = Graph(3, [(0,1),(0,2)])

        self.mask1 = nn.Parameter(graph1.mask, requires_grad = False)
        #self.layer4 = GraphLinearLayer(graph1.vertices, mask1)
        #self.layer5 = GraphLinearLayer(graph1.vertices, mask1)
    
    def forward(self, x):
        #x = F.selu(self.layer3(F.selu(self.layer2(F.selu(self.layer1(F.selu(self.layer0(x))))))))
        x = self.layer1(F.selu(self.layer0(x)))
        x = self.pool(x)
        #x = self.layer5(F.selu(self.layer4(x)))
        return F.linear(x, self.avg)

class FCN(nn.Module):
    
    def __init__(self):
        super(FCN, self).__init__()
        self.layer0 = nn.Linear(8,8, bias=False)
        self.layer1 = nn.Linear(8,8, bias = False)
        #self.layer2 = nn.Linear(8,8)
        #self.layer3 = nn.Linear(8,8)
        self.layer4 = nn.Linear(8,1, bias = False)
        #self.layer5 = nn.Linear(3,3)
        #self.layer6 = nn.Linear(3,3)
	
        #self.avg = nn.Parameter(torch.FloatTensor([[1.0/3, 1.0/3, 1.0/3]]), requires_grad = False)
    def forward(self, x):
        #x = F.selu(self.layer3(F.selu(self.layer2(F.selu(self.layer1(F.selu(self.layer0(x))))))))
        x = self.layer1(self.layer0(x))
        x = self.layer4(x)
        #x = self.layer6(F.selu(self.layer5(x)))
        #return F.linear(x,self.avg)
        return x

def train(data, y, net):
    print(torch.cuda.is_available())
    start = time.time()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    data = [x.cuda() for x in data]
    x = torch.cat(data, 0).view(len(data), -1)
    y = y.view(-1, 1)
    y = y.cuda()
    print(x.size())
    losses = []
    THRESHHOLD= 1E-6
    
    for epoch in range(30000):
        
        total_loss = 0
        optimizer.zero_grad()
        
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
        if (epoch + 1)%200 == 0:
            print(total_loss)
            losses.append(total_loss)
            if total_loss < THRESHHOLD:
                print("FINISHED TRAINING")
                print("Total number of epochs = ", epoch+1)
                print("Total time = ", time.time() - start)
                break
    return losses

def predict(x, net):
    with torch.no_grad():
        x = x.cuda()
        index = net(x).cpu().item() > 0
    return index

net = GCN(Graph(8, edge_list))
fcn = FCN()
bound = 1e-3
for idx, param in enumerate(list(fcn.parameters())):
    init = torch.Tensor(param.size()).uniform_(-bound, bound)
    param.data = init
if torch.cuda.is_available():
     net.cuda()
     fcn.cuda()
loss_gcn = train(data, y, net)
loss_fcn = train(data, y, fcn)
net.eval()
fcn.eval()
results_gcn = [predict(x, net) for x in data]
results_fcn = [predict(x, fcn) for x in data]
test_results_gcn = [predict(x, net) for x in test_data]
test_results_fcn = [predict(x, fcn) for x in test_data]
accuracy_gcn = sum([results_gcn[i] == labels[i] for i in range(len(labels))])/float(len(labels))
accuracy_fcn = sum([results_fcn[i] == labels[i] for i in range(len(labels))])/float(len(labels))

test_accuracy_gcn = sum([test_results_gcn[i] == test_labels[i] for i in range(len(test_labels))])/float(len(test_labels))
test_accuracy_fcn = sum([test_results_fcn[i] == test_labels[i] for i in range(len(test_labels))])/float(len(test_labels))
print("accuracy_gcn = ", accuracy_gcn)
print("accuracy_fcn = ", accuracy_fcn)
print("test_accuracy_gcn = ", test_accuracy_gcn)
print("test_accuracy_fcn = ", test_accuracy_fcn)
print(list(net.parameters()))
print(list(fcn.parameters()))
w_gcn,v_gcn = LA.eig(list(net.cpu().parameters())[4].data.numpy())
w_fcn,v_fcn = LA.eig(list(fcn.cpu().parameters())[0].data.numpy())
print(sorted(w_gcn, key = lambda x: numpy.abs(x)))
print(sorted(w_fcn, key = lambda x: numpy.abs(x)))
