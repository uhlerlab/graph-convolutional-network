import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from IPython import embed


class GCN(nn.Module):

    def __init__(self, vertices, mask):
        super(GCN, self).__init__()

        self.vertices = vertices
        self.mask = Variable(mask, requires_grad = False)

        self.layer1 = nn.Linear(vertices, vertices, bias = False)
        self.layer1.weight.data = self.layer1.weight.data*self.mask.data

        self.layer2 = nn.Linear(vertices, vertices, bias = False)
        self.layer2.weight.data = self.layer2.weight.data*self.mask.data

        self.layer3 = nn.Linear(vertices, vertices, bias = False)
        self.layer3.weight.data = self.layer3.weight.data*self.mask.data


    def forward(self, x):
        self.layer1.weight.data = self.layer1.weight.data*self.mask.data
        self.layer2.weight.data = self.layer2.weight.data*self.mask.data
        self.layer3.weight.data = self.layer3.weight.data*self.mask.data
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def train(data, net):

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(1000):
        total_loss = 0
        for x in data:
           
            optimizer.zero_grad()

            outputs = net(x)
            loss = criterion(outputs, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if (epoch + 1)%100 == 0:
            print(total_loss)

    print('Finished Training')

def create_mask(num_vertices, edge_list):
    mask = torch.zeros(num_vertices,num_vertices)

    for i,j in edge_list:
        mask[i,j] = 1.0
        mask[j,i] = 1.0
    for i in range(num_vertices):
        mask[i,i] = 1.0
    return mask

# num_vertices = 5
# edge_list = [(0,1), (1,2), (2,0), (0,3), (1,4)]
num_vertices = 3
edge_list = [(0,1), (1,2)]

mask = create_mask(num_vertices, edge_list)
net = GCN(num_vertices, mask)

data = [torch.randn(num_vertices, dtype=torch.float) for i in range(10)]

print(list(net.parameters())[0])
train(data,net)
print(list(net.parameters())[0])