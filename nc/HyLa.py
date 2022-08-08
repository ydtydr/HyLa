#!/usr/bin/env python3
# import sys
# sys.path.append("..") 
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import torch
import logging
import argparse
import json
from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
from hype import MANIFOLDS, MODELS, build_model, get_model
from hype.hyla_utils import sgc_precompute, acc_f1, load_data, load_reddit_data
import torch.nn.functional as F
import timeit
import gc
from sklearn.metrics import roc_auc_score, average_precision_score

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

def test_regression(model_f, model_c, features, test_labels, test_index=None, metric='acc'):
    with torch.no_grad():
        model_f.eval()
        model_c.eval()
        HyLa_features = model_f()
        HyLa_features = torch.mm(features.to(HyLa_features.device), HyLa_features)
        predictions = model_c(HyLa_features)
        del HyLa_features
        acc, f1 = acc_f1(predictions, test_labels)
    if metric == 'f1':
        return f1
    return acc

def train(model_f,
          model_c,
          optimizer_f,
          optimizer_c,
          data,
          opt,
          log,
          progress=False,
          ckps=None,
):
    model_f.train()
    model_c.train()
    val_acc_best = 0.0
    train_acc_best = 0.0
    for epoch in range(opt.epoch_start, opt.epochs):
        t_start = timeit.default_timer()
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        HyLa_features = model_f()
        HyLa_features = torch.mm(data['features_train'].to(opt.device), HyLa_features)
        predictions = model_c(HyLa_features)
        del HyLa_features
        loss = F.cross_entropy(predictions, data['labels'][data['idx_train']].to(opt.device))
        loss.backward()
        optimizer_f.step()
        optimizer_c.step()
        train_acc = test_regression(
            model_f, model_c, data['features_train'], data['labels'][data['idx_train']].to(opt.device), metric = opt.metric)
        val_acc = test_regression(model_f, model_c, data['features'][data['idx_val']], 
                                  data['labels'][data['idx_val']].to(opt.device), metric = opt.metric)
        if val_acc>val_acc_best:
            val_acc_best = val_acc
            if ckps is not None:
                ckps[0].save({
                'model': model_f.state_dict(),
                'epoch': epoch,
                'val_acc_best': val_acc_best,
                })
                ckps[1].save({
                'model': model_c.state_dict(),
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
                f'"loss_c": {loss.cpu().item():.4f}, '
                '}'
            ) 
        gc.collect()
        torch.cuda.empty_cache()
    return train_acc, train_acc_best, val_acc, val_acc_best

def main():
    parser = argparse.ArgumentParser(description='Train HyLa-SGC for node classification tasks')
    parser.add_argument('-checkpoint', action='store_true', default=False)
    parser.add_argument('-task', type=str, default='nc', help='learning task')
    parser.add_argument('-dataset', type=str, required=True,
                        help='Dataset identifier [cora|disease_nc|pubmed|citeseer|reddit|airport]')
    parser.add_argument('-he_dim', type=int, default=2,
                        help='Hyperbolic Embedding dimension')
    parser.add_argument('-hyla_dim', type=int, default=100,
                        help='HyLa feature dimension')
    parser.add_argument('-order', type=int, default=2,
                        help='order of adjaceny matrix in SGC precomputation')
    parser.add_argument('-manifold', type=str, default='poincare',
                        choices=MANIFOLDS.keys(), help='model of hyperbolic space')
    parser.add_argument('-model', type=str, default='hyla',
                        choices=MODELS.keys(), help='feature model class, hyla|rff')
    parser.add_argument('-lr_e', type=float, default=0.1,
                        help='Learning rate for hyperbolic embedding')
    parser.add_argument('-lr_c', type=float, default=0.1,
                        help='Learning rate for the classifier SGC')
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
    parser.add_argument('-optim_type', choices=['adam', 'sgd'], default='adam', help='optimizer used for the classification SGC model')
    parser.add_argument('-metric', choices=['acc', 'f1'], default='acc', help='what metrics to report')    
    parser.add_argument('-lambda_scale', type=float, default=0.07, help='scale of lambdas when generating HyLa features')
    parser.add_argument('-inductive', action='store_true', default=False, help='inductive training, used for reddit.')
    parser.add_argument('-use_feats', action='store_true', default=False, help='whether embed in the feature level, otherwise node level')
    parser.add_argument('-tuned', action='store_true', default=False, help='whether use tuned hyper-parameters')
    opt = parser.parse_args()
    
    if opt.tuned:
        with open(f'{currentdir}/hyper_parameters_{opt.he_dim}d.json',) as f:
            hyper_parameters = json.load(f)[opt.dataset]
        opt.he_dim = hyper_parameters['he_dim']
        opt.hyla_dim = hyper_parameters['hyla_dim']
        opt.order = hyper_parameters['order']
        opt.lambda_scale = hyper_parameters['lambda_scale']
        opt.lr_e = hyper_parameters['lr_e']
        opt.lr_c = hyper_parameters['lr_c']
        opt.epochs = hyper_parameters['epochs']
    
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
    opt.device = torch.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    
    # here for loading adj things for classification task
    data_path = f'{currentdir}/datasets/' + opt.dataset + '/'
    if opt.dataset in ['cora', 'disease_nc', 'pubmed', 'citeseer', 'airport']:
        data = load_data(opt, data_path)
    elif opt.dataset in ['reddit']:
        data = load_reddit_data(data_path)
    else:
        raise NotImplemented

    ### setup dataset parameters and setting
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
    if opt.progress:
        log.info(f'nonzero_perc during adjacency matrix precomputations: {nonzero_perc}%')
        
    # build feature models and setup optimizers
    model_f = build_model(opt, feature_dim).to(opt.device)
    if opt.lre_type == 'scale':
        opt.lr_e = opt.lr_e * len(data['idx_train'])
    if opt.manifold == 'euclidean':
#         optimizer_f = torch.optim.Adam(model_f.parameters(), lr=opt.lr_e)# weight_decay=1.3e-5
        optimizer_f = torch.optim.SGD(model_f.parameters(), lr=opt.lr_e)
    elif opt.manifold == 'poincare':
        optimizer_f = RiemannianSGD(model_f.optim_params(), lr=opt.lr_e)
    
    # build classification SGC models and setup optimizers
    model_c = get_model("SGC", opt.hyla_dim, data['labels'].max().item()+1).to(opt.device)
    if opt.optim_type == 'sgd':
        optimizer_c = torch.optim.SGD(model_c.parameters(), lr=opt.lr_c)
    elif opt.optim_type == 'adam':
        optimizer_c = torch.optim.Adam(model_c.parameters(), lr=opt.lr_c)#, weight_decay=1.0e-4)
    else:
        raise NotImplementedError

        
    ckps = None
    if opt.checkpoint:
        # setup checkpoint
        ckp_fm = generate_ckpt(opt, model_f, f'{currentdir}/datasets/' + opt.dataset + '/fm.pt')
        ckp_cm = generate_ckpt(opt, model_c, f'{currentdir}/datasets/' + opt.dataset + '/cm.pt')
        ckps = (ckp_fm, ckp_cm)
    t_start_all = timeit.default_timer()
    train_acc, train_acc_best, val_acc, val_acc_best = train(
        model_f, model_c, optimizer_f, optimizer_c, 
        data, opt, log, progress=opt.progress, ckps=ckps)
    if opt.progress:
        log.info(f'TOTAL ELAPSED: {timeit.default_timer()-t_start_all:.2f}')
    if opt.checkpoint and ckps is not None:
        state_fm = ckps[0].load()
        state_cm = ckps[1].load()
        model_f.load_state_dict(state_fm['model'])
        model_c.load_state_dict(state_cm['model'])
        if opt.progress:
            log.info(f'early stopping, loading from epoch: {state_fm["epoch"]} with val_acc_best: {state_fm["val_acc_best"]}')
    test_acc = test_regression(
        model_f, model_c, data['features'][data['idx_test']], data['labels'][data['idx_test']].to(opt.device), 
        metric = opt.metric)
#     test_acc_threshold = {'cora': 0, 'disease_nc': 0, 'pubmed': 0, 'citeseer': 0, 'reddit': 0, 'airport': 0}
#     test_acc_threshold = {'cora': 82, 'disease_nc': 80, 'pubmed': 80, 'citeseer': 71, 'reddit': 93.5, 'airport': 80}
#     if test_acc * 100.0 > test_acc_threshold[opt.dataset]:
    log.info(
            f'"|| last train_acc": {train_acc*100.0:.2f}%, '
            f'"|| best train_acc": {train_acc_best*100.0:.2f}%, '
            f'"|| last val_acc": {val_acc*100.0:.2f}%, '
            f'"|| best val_acc": {val_acc_best*100.0:.2f}%, '
            f'"|| test_acc": {test_acc*100.0:.2f}%.'
        )

if __name__ == '__main__':
    main()
