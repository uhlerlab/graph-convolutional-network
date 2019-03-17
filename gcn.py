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

    def __init__(self, input_graph):
        super(GCN, self).__init__()

        input_graph = input_graph

        # mask0 = Variable(input_graph.mask, requires_grad = False)
        # self.layer0 = GraphLinearLayer(input_graph.vertices, mask0)
        # contraction0, graph1 = self.coarsen(input_graph)
        # self.pool = GraphPool(input_graph.vertices, contraction0)
        
        # mask1 = Variable(graph1.mask, requires_grad = False) 
        # self.layer1 = GraphLinearLayer(graph1.vertices, mask1)

        # self.reverse_pool = GraphReversePool(input_graph.vertices, contraction0)
        # self.layer2 = GraphLinearLayer(input_graph.vertices, mask0)

        mask0 = Variable(input_graph.mask, requires_grad = False)
        self.layer0 = GraphLinearLayer(input_graph.vertices, mask0)
        contraction0 = [[0,1,2,3],[2,3,4,5]]
        self.pool = GraphPool(input_graph.vertices, contraction0)
        graph1 = Graph(2, [(0,1)])

        mask1 = Variable(graph1.mask, requires_grad = False) 
        self.layer1 = GraphLinearLayer(graph1.vertices, mask1)

        self.reverse_pool = GraphReversePool(input_graph.vertices, contraction0)
        self.layer2 = GraphLinearLayer(input_graph.vertices, mask0)

    def encode(self, x):
        x = F.leaky_relu(self.layer0(x))
        x = self.pool(x)
        x = F.leaky_relu(self.layer1(x))
        # x = self.layer1(x)

        return x

    def decode(self, x):
        x = self.reverse_pool(x)
        x = self.layer2(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def coarsen(self, graph):
        coarsening = []
        temp_mask = graph.mask.clone()
        mapping = {}
        num_found = 0
        while num_found < graph.vertices:
            _value, index = torch.sum(temp_mask, dim=1).max(0)
            group = [i for i in range(graph.vertices) if temp_mask[i, index] == 1]
            for i in group:
                mapping[i] = len(coarsening)
            num_found += len(group)
            coarsening.append(group)
            for i in group:
                temp_mask[i] = torch.zeros(graph.vertices)
                torch.transpose(temp_mask,0,1)[i] = torch.zeros(graph.vertices)

        new_edge_list = set()
        for i, j in graph.edge_list:

            i_new = mapping[i]
            j_new = mapping[j]
            if i_new != j_new:
                new_edge_list.add((min([i_new,j_new]), max([i_new,j_new])))

        new_graph = Graph(len(coarsening), list(new_edge_list))

        return coarsening, new_graph


class Graph:

    def __init__(self, vertices, edge_list):
        self.vertices = vertices
        self.edge_list = edge_list
        self.mask = self.create_mask(vertices, edge_list)

    def create_mask(self, num_vertices, edge_list):
        mask = torch.zeros(num_vertices,num_vertices)

        for i,j in edge_list:
            mask[i,j] = 1.0
            mask[j,i] = 1.0
        for i in range(num_vertices):
            mask[i,i] = 1.0
        return mask


def train(data, net):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    # optimizer = optim.SGD(net.parameters(), lr=1e-1)
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

if __name__ == "__main__":
    num_vertices = 5
    edge_list = [(0,1), (1,2), (2,0), (0,3), (1,4)]

    net = GCN(Graph(num_vertices,edge_list))

    data = [torch.rand(num_vertices, dtype=torch.float) for i in range(1)]

    print(list(net.parameters())[0])
    train(data,net)
    print(list(net.parameters())[0])