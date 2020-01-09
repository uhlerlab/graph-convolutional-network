import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BaseLayer(nn.Module):
    def __init__(self, n, init_val):
        
        super(BaseLayer, self).__init__()
        self.n = n
        self.init_val = init_val
        self.init = self.init_val * torch.randn(n, n)
        self.weight = nn.Parameter(torch.tensor(self.init))
        
    def forward(self, input):
        return F.linear(input, self.weight)

    def weight_matrix(self):
    	return self.weight

class DeepNet(nn.Module):
    def __init__(self, n, depth):
        
        super(DeepNet, self).__init__()
        self.n = n
        self.depth = depth
        self.init_val = 1e-1
        self.layers = nn.ModuleList([BaseLayer(n, self.init_val) for i in range(depth)])

    def forward(self, input):
        x = input
        for i in range(self.depth):
            x = self.layers[i](x)
        return x
    
    def weight_matrix(self):
        net = torch.eye(self.n)
        for i in range(self.depth):
            net = torch.mm(self.layers[i].weight_matrix(), net)
        return net

def train(net, x, threshhold):
    optimizer = optim.SGD(net.parameters(), lr = 0.1)
    criterion = nn.MSELoss()
    init = net.weight_matrix()
    svals = []
    loss_val = 1.0
    losses = []
    i = 0
    align = []
    while loss_val > threshhold:
        align.append(alignment(net, x))
        svals.append(np.linalg.svd(net.weight_matrix().detach().numpy())[1])
        optimizer.zero_grad()
        y = net(x)
        loss = criterion(x, y)
        loss.backward()
        optimizer.step()
        loss_val = loss.data.item()
        losses.append(loss_val)
        if i % 1000 == 0:
            print(loss_val)
        i+=1
    return losses, align, svals

def alignment(net, x):

    x_unit = x / np.linalg.norm(x)
    x_unit = x_unit.reshape(-1)
    u_vals = []
    vh_vals = []
    for layer in net.layers:
        u, s, vh = np.linalg.svd(layer.weight_matrix().detach().numpy())
        u_vals.append(u)
        vh_vals.append(vh)
    
    w0 = net.layers[0].weight_matrix().detach().numpy()
    res = np.matmul(np.matmul(w0.transpose() , w0), x_unit.reshape(-1, 1))
    total_alignment = abs((np.dot(res.reshape(-1), x_unit.reshape(-1))/np.linalg.norm(res)).item())
    wn = net.layers[len(net.layers) - 1].weight_matrix().detach().numpy()
    res = np.matmul(np.matmul(wn , wn.transpose()), x_unit.reshape(-1, 1))
    total_alignment += abs((np.dot(res.reshape(-1), x_unit.reshape(-1))/np.linalg.norm(res)).item())
    
    if len(net.layers) > 1:
        for i in range(len(net.layers) - 1):
            wi = net.layers[i].weight_matrix().detach().numpy()
            res = np.matmul(np.matmul(wi , wi.transpose()), vh_vals[i+1][0].reshape(-1, 1))
            total_alignment += abs((np.dot(res.reshape(-1), vh_vals[i+1][0])/np.linalg.norm(res)).item())
    return total_alignment/float(len(net.layers) + 1)

# def alignment(net, x):

#     x_unit = x / np.linalg.norm(x)
#     x_unit = x_unit.reshape(-1)
#     u_vals = []
#     vh_vals = []
#     for layer in net.layers:
#         u, s, vh = np.linalg.svd(layer.weight_matrix().detach().numpy())
#         u_vals.append(u)
#         vh_vals.append(vh)
    
#     w0 = net.layers[0].weight_matrix().detach().numpy()
#     res = np.matmul(np.matmul(w0.transpose() , w0), x_unit.reshape(-1, 1))
#     total_alignment = abs((np.dot(res.reshape(-1), x_unit.reshape(-1))/np.linalg.norm(res)).item())
#     wn = net.layers[len(net.layers) - 1].weight_matrix().detach().numpy()
#     res = np.matmul(np.matmul(wn , wn.transpose()), x_unit.reshape(-1, 1))
#     total_alignment += abs((np.dot(res.reshape(-1), x_unit.reshape(-1))/np.linalg.norm(res)).item())
    
    if len(net.layers) > 1:
            
        total_alignment += sum([abs(np.dot(u_vals[i].transpose()[0], vh_vals[i+1][0])) 
            for i in range(len(net.layers) - 1)])
    return total_alignment/float(len(net.layers) + 1)

def stable_rank(svals):
    squares = [i**2 for i in svals]
    return sum(squares)/max(squares)
