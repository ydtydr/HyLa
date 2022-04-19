import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
import numpy as np
from .hyla_utils import PoissonKernel, sample_boundary, measure_tensor_size


class LaplacianNN(nn.Module):
    def __init__(self, manifold, dim, size, HyLa_fdim, scale=0.1, sparse=False, **kwargs):
        super(LaplacianNN, self).__init__()
        self.manifold = manifold
        self.lt = manifold.allocate_lt(size, dim, sparse)
        self.manifold.init_weights(self.lt)
        self.dim = dim
        self.Lambdas = scale * torch.randn(HyLa_fdim)
        self.boundary = sample_boundary(HyLa_fdim, self.dim, cls='RandomUniform')
        self.bias = 2 * np.pi * torch.rand(HyLa_fdim)
    
    def forward(self):
        with torch.no_grad():
            e_all = self.manifold.normalize(self.lt.weight)
        PsK = PoissonKernel(e_all, self.boundary.to(e_all.device))
        angles = self.Lambdas.to(e_all.device)/2.0 * torch.log(PsK)
        eigs = torch.cos(angles + self.bias.to(e_all.device)) * torch.sqrt(PsK)**(self.dim-1)
        return eigs
    
    def optim_params(self):
        return [{
            'params': self.lt.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]
    
    
class EucLaplacianNN(nn.Module):
    def __init__(self, manifold, dim, size, HyLa_fdim, scale=0.1, sparse=False, **kwargs):
        super(EucLaplacianNN, self).__init__()
        self.manifold = manifold
        self.lt = manifold.allocate_lt(size, dim, sparse)
        self.manifold.init_weights(self.lt)
        self.norm = 1. / np.sqrt(dim)
        self.Lambdas = nn.Parameter(torch.from_numpy(np.random.normal(loc=0, scale=scale, size=(dim, HyLa_fdim))), requires_grad=False) 
        self.bias = nn.Parameter(torch.from_numpy(np.random.uniform(0, 2 * np.pi, size=HyLa_fdim)),requires_grad=False)
    
    def forward(self):
        with torch.no_grad():
            e_all = self.manifold.normalize(self.lt.weight)
        features = self.norm * np.sqrt(2) * torch.cos(e_all @ self.Lambdas + self.bias)
        return features
    
    def optim_params(self):
        return [{
            'params': self.lt.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)
    
class GraphConvolution(nn.Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, adj, bias=True, use_linear=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        self.use_linear = use_linear
        if use_linear:
            self.W = nn.Linear(in_features, out_features, bias=bias)
            self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
#         support = torch.spmm(self.adj.to(input.device), input)
#         output = self.W(support)
        ####################
        if self.use_linear:
            support = self.W(input)
        else:
            support = input
        output = torch.spmm(self.adj.to(support.device), support)
#         output = torch.mm(self.adj.to(support.device), support)
        return output

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nclass, adj, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = []
        for layer_idx in range(len(nfeat)-1):
            self.layers.append(GraphConvolution(nfeat[layer_idx], nfeat[layer_idx+1], adj, use_linear=False).cuda())
#             self.layers.append(GraphConvolution(nfeat[layer_idx], nfeat[layer_idx+1], adj).cuda())
#         self.gc_class = GraphConvolution(nfeat[-1], nclass, adj).cuda()
        self.gc_class = GraphConvolution(nfeat[0], nclass, adj, use_linear=True).cuda()
        self.dropout = dropout

    def forward(self, x, use_relu=True):
        for layer in self.layers:
            x = layer(x)
            if use_relu:
                x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc_class(x)
#         raise
        return x#[inputs]
    
class MLP(nn.Module):
    def __init__(self, nfeat, nclass, adj, dropout=0.5):
        super(MLP, self).__init__()
        self.adj = adj
        self.layers = []
        for layer_idx in range(len(nfeat)-1):
            self.layers.append(nn.Linear(nfeat[layer_idx], nfeat[layer_idx+1]).cuda())
        self.gc_class = nn.Linear(nfeat[-1], nclass).cuda()
        self.dropout = dropout

    def forward(self, x, inputs, use_relu=True):
        for layer in self.layers:
            x = layer(x)
            if use_relu:
                x = F.relu(x)
#             x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc_class(x)
#         output = torch.spmm(self.adj.to(support.device), support)
        x = torch.mm(self.adj.to(x.device), x)
        return x[inputs]
