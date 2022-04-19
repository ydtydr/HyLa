#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import manifolds
from . import energy_function
import argparse
from . import models

MANIFOLDS = {
    'lorentz': manifolds.LorentzManifold,
    'poincare': manifolds.PoincareManifold,
    'euclidean': manifolds.EuclideanManifold,
}

MODELS = {
    'distance': energy_function.DistanceEnergyFunction,
    'entailment_cones': energy_function.EntailmentConeEnergyFunction,
    'laplaNN': models.LaplacianNN,
    'EuclaplaNN': models.EucLaplacianNN,
}

def build_model(opt, N):
    if isinstance(opt, argparse.Namespace):
        opt = vars(opt)
    K = 0.1 if opt['model'] == 'entailment_cones' else None
    manifold = MANIFOLDS[opt['manifold']](K=K)
    return MODELS[opt['model']](
        manifold,
        dim=opt['dim'],
        size=N,
        HyLa_fdim=opt['HyLa_fdim'],
        scale=opt['scale'],
        sparse=opt['sparse'],
    )

def get_model(model_opt, nfeat, nclass, adj=None, dropout=0.0):
    if model_opt == "GCN":
        assert adj is not None
        model = models.GCN(nfeat=nfeat,
                    nclass=nclass,
                    adj=adj,
                    dropout=dropout)
    elif model_opt == "SGC":
        model = models.SGC(nfeat=nfeat,
                    nclass=nclass)
    elif model_opt == "MLP":
        model = models.MLP(nfeat=nfeat,
                    nclass=nclass,
                           adj=adj)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))
    return model
