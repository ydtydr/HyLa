#!/usr/bin/env python3
import numpy as np
import torch
import logging
import argparse
from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
import sys
import json
from hype import MANIFOLDS, MODELS, build_model, get_model
from hype.hyla_utils import sgc_precompute, acc_f1, load_data, load_reddit_data
import torch.nn.functional as F
import timeit
import gc
from sklearn.metrics import roc_auc_score, average_precision_score
import json


def generate_ckpt(opt, model, path):
    checkpoint = LocalCheckpoint(
            path,
            include_in_all={'conf' : vars(opt)},
            start_fresh=opt.fresh
        )
    # get state from checkpoint
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    opt.epoch_start = state['epoch']
    return checkpoint

def test_regression(feature_model, class_model, features, test_labels, test_index=None, cmodel='sgc', metric='acc'):
    with torch.no_grad():
        feature_model.eval()
        class_model.eval()
        HyLa_features = feature_model()
        if cmodel == 'sgc':
            HyLa_features = torch.mm(features.to(HyLa_features.device), HyLa_features)
            predictions = class_model(HyLa_features)
        elif cmodel == 'gcn':
            HyLa_features = torch.spmm(features.to(HyLa_features.device), HyLa_features)
            predictions = class_model(HyLa_features)[test_index]
        elif cmodel == 'mlp':
            HyLa_features = torch.spmm(features.to(HyLa_features.device), HyLa_features)
            predictions = class_model(HyLa_features, test_index)
        else:
            raise NotImplementedError
        del HyLa_features
        acc, f1 = acc_f1(predictions, test_labels)
    if metric == 'f1':
        return f1
    return acc

def train(device,
          model,
          class_model,
          optimizers,
          optimizer_class,
          data,
          opt,
          log,
          progress=False,
          ckps=None,
):
    model.train()
    class_model.train()
    val_acc_best = 0.0
    val_acc = 0.0
    train_acc = 0.0
    train_acc_best = 0.0
    optimizer_emb = optimizers[0]
    for epoch in range(opt.epoch_start, opt.epochs):
        t_start = timeit.default_timer()
        optimizer_emb.zero_grad()
        optimizer_class.zero_grad()
        HyLa_features = model()
        if opt.cmodel == 'sgc':
            HyLa_features = torch.mm(data['features_train'].to(HyLa_features.device), HyLa_features)
            predictions = class_model(HyLa_features)
        else:
            ### only for gcn rn
            HyLa_features = torch.spmm(data['features'].to(HyLa_features.device), HyLa_features)
            predictions = class_model(HyLa_features)[data['idx_train']]
        del HyLa_features
        loss_class = F.cross_entropy(predictions, data['labels'][data['idx_train']].to(device))
        loss_class.backward()
        optimizer_class.step()
        optimizer_emb.step()
        if opt.cmodel == 'sgc':
            train_acc = test_regression(
                model, class_model, data['features_train'], data['labels'][data['idx_train']].to(device),
                cmodel=opt.cmodel, metric = opt.metric)
            val_acc = test_regression(
                model, class_model, data['features'][data['idx_val']], data['labels'][data['idx_val']].to(device), 
                cmodel=opt.cmodel, metric = opt.metric)
        else:# for gcn rn
            train_acc = test_regression(
                model, class_model, data['features'], 
                data['labels'][data['idx_train']].to(device), test_index=data['idx_train'],
                cmodel=opt.cmodel, metric = opt.metric)
            val_acc = test_regression(
                model, class_model, data['features'], 
                data['labels'][data['idx_val']].to(device), test_index=data['idx_val'], 
                cmodel=opt.cmodel, metric = opt.metric)
        if val_acc>val_acc_best:
            val_acc_best = val_acc
            if ckps is not None:
                ckps[0].save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc_best': val_acc_best,
                })
                ckps[1].save({
                'model': class_model.state_dict(),
                'epoch': epoch,
                'val_acc_best': val_acc_best,
                })
        if train_acc>train_acc_best:
            train_acc_best = train_acc
        if progress:
            log.info(
                'running stats: {'
                f'"epoch": {epoch}, '
                f'"elapsed": {timeit.default_timer()-t_start:.2f}, '
                f'"train_acc": {train_acc*100.0:.2f}%, '
                f'"val_acc": {val_acc*100.0:.2f}%, '
                f'"loss_c": {loss_class.cpu().item():.4f}, '
                '}'
            )
                
        gc.collect()
        torch.cuda.empty_cache()
    return train_acc, train_acc_best, val_acc, val_acc_best
    
def train_reddit_lbfgs(device,
          model,
          class_model,
          optimizer_emb,
          optimizer_class,
          data,
          opt,
          log,
          progress=False,
          ckps=None,
):
    model.train()
    class_model.train()
    val_acc_best = 0.0
    train_acc_best = 0.0
    for epoch in range(opt.epoch_start, opt.epochs):
        t_start = timeit.default_timer()
        optimizer_emb.zero_grad()
        HyLa_features = model()
        HyLa_features = torch.mm(data['features_train'].to(HyLa_features.device), HyLa_features)
        predictions = class_model(HyLa_features)    
        loss_class = F.cross_entropy(predictions, data['labels'][data['idx_train']].to(device))
        loss_class.backward()
        optimizer_emb.step()
        def closure():
            optimizer_class.zero_grad()
            HyLa_features = model()
            HyLa_features = torch.mm(data['features_train'].to(HyLa_features.device), HyLa_features)
            predictions = class_model(HyLa_features)
            loss_class = F.cross_entropy(predictions, data['labels'][data['idx_train']].to(device))
            loss_class.backward()
            return loss_class
        loss_class = optimizer_class.step(closure)
        ########## debug gradients ###########
#         print(model.lt.weight.grad)
#         print(class_model.W.weight.grad)
#         raise
        ########## debug gradients ###########
        train_acc = test_regression(
            model, class_model, data['features_train'], data['labels'][data['idx_train']].to(device),
            cmodel=opt.cmodel, metric = opt.metric)
        val_acc = test_regression(
            model, class_model, data['features'][data['idx_val']], data['labels'][data['idx_val']].to(device), 
            cmodel=opt.cmodel, metric = opt.metric)
        if val_acc>val_acc_best:
            val_acc_best = val_acc
            if ckps is not None:
                ckps[0].save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc_best': val_acc_best,
                })
                ckps[1].save({
                'model': class_model.state_dict(),
                'epoch': epoch,
                'val_acc_best': val_acc_best,
                })
        if train_acc>train_acc_best:
            train_acc_best = train_acc
        if progress:
            log.info(
                'running stats: {'
                f'"epoch": {epoch}, '
                f'"elapsed": {timeit.default_timer()-t_start:.2f}, '
                f'"train_acc": {train_acc*100.0:.2f}%, '
                f'"val_acc": {val_acc*100.0:.2f}%, '
                f'"loss_c": {loss_class.cpu().item():.4f}, '
                '}'
            )
                
        gc.collect()
        torch.cuda.empty_cache()
    return train_acc, train_acc_best, val_acc, val_acc_best

def main():
    parser = argparse.ArgumentParser(description='Train Laplacian Net')
    parser.add_argument('-checkpoint', action='store_true', default=False)
    parser.add_argument('-task', type=str, required=True,
                        help='task [nc|lp]')
    parser.add_argument('-dataset', type=str, required=True,
                        help='Dataset identifier [cora|disease_nc|pubmed|citeseer|reddit|airport]')
    parser.add_argument('-dim', type=int, default=2,
                        help='Hyperbolic Embedding dimension')
    parser.add_argument('-HyLa_fdim', type=int, default=100,
                        help='Laplacian feature dimension')
    parser.add_argument('-order', type=int, default=5,
                        help='order of adjaceny matrix')
    parser.add_argument('-manifold', type=str, default='poincare',
                        choices=MANIFOLDS.keys())
    parser.add_argument('-model', type=str, default='laplaNN',
                        choices=MODELS.keys(), help='model class')
    parser.add_argument('-cmodel', type=str, default='gcn',
                        choices=['sgc', 'gcn', 'mlp'], help='classification model class')
    parser.add_argument('-lr_e', type=float, default=0.1,
                        help='Learning rate for the embedding')
    parser.add_argument('-lr_c', type=float, default=0.1,
                        help='Learning rate for the classifier')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('-strategy', type=int, default=0,
                        help='Epochs of burn in, some advanced definition')
    parser.add_argument('-eval_each', type=int, default=1,
                        help='Run evaluation every n-th epoch')
    parser.add_argument('-fresh', action='store_true', default=False,
                        help='Override checkpoint')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='Print debuggin output')
    parser.add_argument('-gpu', default=0, type=int,
                        help='Which GPU to run on (-1 for no gpu)')
    parser.add_argument('-seed', default=43, type=int, help='random seed')
    parser.add_argument('-sparse', default=True, action='store_true',
                        help='Use sparse gradients for embedding table')
    parser.add_argument('-quiet', action='store_true', default=False)
    parser.add_argument('-lre_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-optim_type', choices=['adam', 'sgd', 'lbfgs'], default='sgd', help='optimizer used for the classification model')
    parser.add_argument('-metric', choices=['acc', 'f1'], default='acc', help='what metrics to report')    
    parser.add_argument('-scale', type=float, default=0.07, help='scale of lambdas when generating Laplacian features')
    parser.add_argument('-inductive', action='store_true', default=False, help='inductive training, used for reddit.')
    parser.add_argument('-use_feats', action='store_true', default=False, help='whether embed in the feature level, otherwise node level')
    opt = parser.parse_args()
    
    # this can be loaded in a more elegant way
    with open('hyper_parameters.json',) as f:
        hyper_parameters = json.load(f)[opt.dataset]
    opt.dim = hyper_parameters['dim']
    opt.HyLa_fdim = hyper_parameters['HyLa_fdim']
    opt.order = hyper_parameters['order']
    opt.scale = hyper_parameters['scale']
    opt.lr_e = hyper_parameters['lr_e']
    opt.lr_c = hyper_parameters['lr_c']
    opt.epoch = hyper_parameters['epoch']
    opt.metric = 'f1' if opt.dataset =='reddit' else 'acc'

    opt.epoch_start = 0
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    opt.split_seed = opt.seed
    opt.progress = not opt.quiet

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger('HyLa')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    # set default tensor type
    torch.set_default_tensor_type('torch.DoubleTensor')
    # set device
    device = torch.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    
    # here for loading adj things for classification task
    data_path = './nc/' + opt.dataset + '/'
    if opt.dataset in ['cora', 'disease_nc', 'pubmed', 'citeseer', 'airport']:
        data = load_data(opt, data_path)
    elif opt.dataset in ['reddit']:
        data = load_reddit_data(data_path)
    else:
        raise NotImplemented

    if opt.use_feats or opt.inductive:
        if opt.progress:
            log.info(f'hyperbolic Laplacian features used in the feature level ...')
        feature_dim = data['features'].size(1)
    else:
        if opt.progress:
            log.info(f'hyperbolic Laplacian features used in the node level ...')
        feature_dim = data['adj_train'].size(1)
    if opt.progress:
        log.info(f'info about the data, training set size :{len(data["idx_train"])}, val size:{len(data["idx_val"])}, test size: {len(data["idx_test"])}')
        log.info(f'size of original feature matrix: {data["features"].size()}, number of classes {data["labels"].max().item()+1}')
    # build models and setup optimizers
    model = build_model(opt, feature_dim).to(device)
    
    indim = opt.HyLa_fdim 
    if opt.cmodel == 'sgc':
        if opt.progress:
            log.info('precomputing features')
        if opt.inductive:
            features = data['features']
            data['features'], _ = sgc_precompute(data['adj_all'], features, opt.order)
            data['features_train'], nonzero_perc = sgc_precompute(data['adj_train'], features[data['idx_train']], opt.order)
        else:
            if not opt.use_feats:
                features = data['adj_train'].to_dense()
                data['features'], nonzero_perc = sgc_precompute(data['adj_train'], features, opt.order-1)
            else:
                features = data['features'].to_dense()        
                data['features'], nonzero_perc = sgc_precompute(data['adj_train'], features, opt.order)
            data['features_train'] = data['features'][data['idx_train']]
        class_model = get_model("SGC", indim, data['labels'].max().item()+1).to(device)
        if opt.progress:
            log.info(f'nonzero_perc: {nonzero_perc}%')
    elif opt.cmodel == 'gcn':
        if not opt.use_feats:
            raise NotImplemented
        class_model = get_model("GCN", [indim, 128, 64, 32, 16], data['labels'].max().item()+1, adj=data['adj_train']).to(device)
    elif opt.cmodel == 'mlp':
        adj_n, nonzero_perc = sgc_precompute(data['adj_train'], torch.eye(data['adj_train'].size(0)), opt.order)
        if opt.progress:
            log.info(f'nonzero_perc: {nonzero_perc}%')
        class_model = get_model("MLP", [indim, 64, 16], data['labels'].max().item()+1, adj=adj_n).to(device)
    else:
        raise NotImplementedError
        
    if opt.progress:
        # Build config string for log
        log.info(f'json_conf: {json.dumps(vars(opt))}')

    if opt.lre_type == 'scale':
        opt.lr_e = opt.lr_e * len(data['idx_train'])
        
    if opt.manifold == 'euclidean':
#         optimizer_emb = torch.optim.Adam(model.parameters(), lr=opt.lr_e)# weight_decay=1.3e-5
        optimizer_emb = torch.optim.SGD(model.parameters(), lr=opt.lr_e)
    elif opt.manifold == 'poincare':
        optimizer_emb = RiemannianSGD(model.optim_params(), lr=opt.lr_e)
        optimizers = [optimizer_emb]
    
    if opt.optim_type == 'sgd':
        optimizer_class = torch.optim.SGD(class_model.parameters(), lr=opt.lr_c)
    elif opt.optim_type == 'adam':
        optimizer_class = torch.optim.Adam(class_model.parameters(), lr=opt.lr_c)#, weight_decay=1.0e-4)
    elif opt.optim_type == 'lbfgs':
        optimizer_class = torch.optim.LBFGS(class_model.parameters(), lr=opt.lr_c)
    else:
        raise NotImplementedError
    
    ckps = None
    if opt.checkpoint:
        # setup checkpoint
        ckp_fm = generate_ckpt(opt, model, './nc/' + opt.dataset + '/fm.pt')
        cls_m = generate_ckpt(opt, model, './nc/' + opt.dataset + '/clm.pt')
        ckps = (ckp_fm, cls_m)
    if opt.optim_type == 'lbfgs':
        train_acc, train_acc_best, val_acc, val_acc_best = train_reddit_lbfgs(
        device, model, class_model, optimizer_emb, optimizer_class, 
        data, opt, log, progress=opt.progress, ckps=ckps)
    else:
        t_start_all = timeit.default_timer()
        train_acc, train_acc_best, val_acc, val_acc_best = train(
            device, model, class_model, optimizers, optimizer_class, 
            data, opt, log, progress=opt.progress, ckps=ckps)
        if opt.progress:
            log.info(f'TOTAL ELAPSED: {timeit.default_timer()-t_start_all:.2f}')
    if opt.checkpoint and ckps is not None:
        state_fm = ckps[0].load()
        state_clm = ckps[1].load()
        model.load_state_dict(state_fm['model'])
        class_model.load_state_dict(state_clm['model'])
        if opt.progress:
            log.info(f'early stopping? loading from epoch: {state_fm["epoch"]} with val_acc_best: {state_fm["val_acc_best"]}')
    if opt.cmodel == 'sgc':
        test_acc = test_regression(
            model, class_model, data['features'][data['idx_test']], data['labels'][data['idx_test']].to(device),
            cmodel=opt.cmodel, metric = opt.metric)
    else: # for gcn rn
        test_acc = test_regression(
                    model, class_model, data['features'], 
                    data['labels'][data['idx_test']].to(device), test_index=data['idx_test'], 
                    cmodel=opt.cmodel, metric = opt.metric)
    test_acc_threshold = {'cora': 75, 'disease_nc': 80, 'pubmed': 70, 'citeseer': 70, 'reddit': 92, 'airport': 90}
    if test_acc * 100.0 > test_acc_threshold[opt.dataset]:
        log.info(
                f'"|| last train_acc": {train_acc*100.0:.2f}%, '
                f'"|| best train_acc": {train_acc_best*100.0:.2f}%, '
                f'"|| last val_acc": {val_acc*100.0:.2f}%, '
                f'"|| best val_acc": {val_acc_best*100.0:.2f}%, '
                f'"|| test_acc": {test_acc*100.0:.2f}%.'
            )

if __name__ == '__main__':
    main()
