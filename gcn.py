import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from IPython import embed
import math

class GraphLinearLayer(nn.Module):
    def __init__(self, vertices, mask):
        super(GraphLinearLayer, self).__init__()
        self.vertices = vertices
        self.mask = Variable(mask, requires_grad = False)
        self.weight = nn.Parameter(torch.Tensor(vertices, vertices))

        #same initialization that pytorch linear layer uses
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 

        self.weight.data = self.weight.data*self.mask.data

    def forward(self, input):
        return F.linear(input, self.weight*self.mask, bias=None)

class GraphPool(nn.Module):
    def __init__(self, vertices, coarsening):
        super(GraphPool, self).__init__()
        self.vertices = vertices
        self.coarsening = coarsening
        self.pool_matrix = self.make_matrix(coarsening)

    def make_matrix(self, coarsening):
        matrix = torch.zeros(len(coarsening), self.vertices)
        for i in range(len(coarsening)):
            for j in coarsening[i]:
                matrix[i,j] = 1.0/float(len(coarsening[i]))
        return Variable(matrix, requires_grad = False)


    def forward(self, input):
        return F.linear(input, self.pool_matrix, bias = None)

class GraphReversePool(nn.Module):
    def __init__(self, vertices, coarsening):
        super(GraphReversePool, self).__init__()
        self.vertices = vertices
        self.coarsening = coarsening
        self.pool_matrix = self.make_matrix(coarsening)

    def make_matrix(self, coarsening):
        matrix = torch.zeros(self.vertices, len(coarsening))
        for j in range(len(coarsening)):
            for i in coarsening[j]:
                matrix[i,j] = 1.0
        return Variable(matrix, requires_grad = False)


    def forward(self, input):
        return F.linear(input, self.pool_matrix, bias = None)

class GCN(nn.Module):

    def __init__(self, vertices, edge_list):
        super(GCN, self).__init__()

        self.vertices = vertices
        self.mask = Variable(self.create_mask(vertices, edge_list), requires_grad = False)
        self.contraction = [[0,1], [2,4], [3]] # hardcoded for now
        self.mask2 = Variable(self.create_mask(3, [(0,1), (0,2)]), requires_grad = False) 

        self.layer1 = GraphLinearLayer(vertices, self.mask)
        self.pool = GraphPool(vertices, self.contraction)
        self.layer2 = GraphLinearLayer(3, self.mask2) 
        self.reverse_pool = GraphReversePool(5, self.contraction)
        self.layer3 = GraphLinearLayer(vertices, self.mask)

    def create_mask(self, num_vertices, edge_list):
        mask = torch.zeros(num_vertices,num_vertices)

        for i,j in edge_list:
            mask[i,j] = 1.0
            mask[j,i] = 1.0
        for i in range(num_vertices):
            mask[i,i] = 1.0
        return mask


    def forward(self, x):
        
        x = F.leaky_relu(self.layer1(x))
        x = self.pool(x)
        x = F.leaky_relu(self.layer2(x))
        x = self.reverse_pool(x)
        x = self.layer3(x)
        return x

def train(data, net):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    for epoch in range(5000):
        total_loss = 0
        for x in data:
           
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.numpy()
        if (epoch + 1)%200 == 0:
            print(total_loss)

    print('Finished Training')


num_vertices = 5
edge_list = [(0,1), (1,2), (2,0), (0,3), (1,4)]
# num_vertices = 3
# edge_list = [(0,1), (1,2)]

# mask = create_mask(num_vertices, edge_list)
net = GCN(num_vertices, edge_list)

data = [torch.rand(num_vertices, dtype=torch.float) for i in range(10)]

print(list(net.parameters())[0])
train(data,net)
print(list(net.parameters())[0])
embed()