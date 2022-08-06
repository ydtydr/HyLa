import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
import numpy as np
from .hyla_utils import PoissonKernel, sample_boundary, measure_tensor_size


class HyLa(nn.Module):
    def __init__(self, manifold, dim, size, HyLa_fdim, scale=0.1, sparse=False, **kwargs):
        super(HyLa, self).__init__()
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
    
    
class RFF(nn.Module):
    def __init__(self, manifold, dim, size, HyLa_fdim, scale=0.1, sparse=False, **kwargs):
        super(RFF, self).__init__()
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