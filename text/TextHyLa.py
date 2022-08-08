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
from hype.hyla_utils import sgc_precompute, sgc_precompute_text, sgc_precompute_text_v1, acc_f1, load_corpus, sparse_to_torch_sparse, sparse_to_torch_dense
import torch.nn.functional as F
import timeit
import gc
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle as pkl

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

def test_regression(model_f, model_c, features, test_labels, metric='acc'):
    with torch.no_grad():
        model_f.eval()
        model_c.eval()
        HyLa_features = model_f()
#         HyLa_features = features.to(HyLa_features.device)
        HyLa_features = torch.mm(features.to(HyLa_features.device), HyLa_features)
        predictions = model_c(HyLa_features)
        del HyLa_features
        if metric == 'mr':
            predict_class = torch.sigmoid(predictions.squeeze()).gt(0.5).float()
            correct = torch.eq(predict_class, test_labels).long().sum().item()
            acc = correct/predict_class.size(0)
        else:
            acc, f1 = acc_f1(predictions, test_labels)
    if metric == 'f1':
        return f1
    return acc

def train(model_f,
          model_c,
          optimizer_f,
          optimizer_c,
          feat_dict,
          label_dict,
          opt,
          log,
          progress=False,
          ckps=None,
):
    model_f.train()
    model_c.train()
    val_acc_best = 0.0
    train_acc_best = 0.0
    feat_train = feat_dict['train'].to(opt.device)
    for epoch in range(opt.epoch_start, opt.epochs):
        t_start = timeit.default_timer()
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        HyLa_features = model_f()
        HyLa_features = torch.mm(feat_train, HyLa_features)
#         HyLa_features = feat_train
        predictions = model_c(HyLa_features)
        del HyLa_features
        if opt.dataset == 'mr':
            loss = F.binary_cross_entropy(torch.sigmoid(predictions.squeeze()), label_dict['train'].to(opt.device))
        else:
            loss = F.cross_entropy(predictions, label_dict['train'].to(opt.device))
        loss.backward()
        ########## debug gradients ###########
#         print(model.lt.weight.grad)
# # #         print(class_model.layers[0].W.weight.grad)
#         print(class_model.W.weight.grad)
#         raise
#         print(model.Lambdas)
#         print(model.Lambdas.grad)
        ########## debug gradients ###########
        optimizer_f.step()
        optimizer_c.step()
        train_acc = test_regression(
            model_f, model_c, feat_dict['train'], label_dict['train'].to(opt.device), metric = opt.metric)
        val_acc = test_regression(
            model_f, model_c, feat_dict['val'], label_dict['val'].to(opt.device), metric = opt.metric)
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
    parser = argparse.ArgumentParser(description='Train HyLa-SGC|HyLa-LR for text classification tasks')
    parser.add_argument('-checkpoint', action='store_true', default=False)
    parser.add_argument('-task', type=str, default='text', help='learning task')
    parser.add_argument('-dataset', type=str, required=True,
                        help='Dataset identifier [20ng|R8|R52|ohsumed|mr]')
    parser.add_argument('-he_dim', type=int, default=2,
                        help='Hyperbolic Embedding dimension')
    parser.add_argument('-hyla_dim', type=int, default=100,
                        help='HyLa feature dimension')
    parser.add_argument('-order', type=int, default=1,
                        help='order of adjaceny matrix in SGC precomputation, only for transductive experiments')
    parser.add_argument('-manifold', type=str, default='poincare',
                        choices=['poincare', 'euclidean', 'none'], help='model of hyperbolic space')   
    parser.add_argument('-model', type=str, default='hyla',
                        choices=['hyla', 'rff', 'none'], help='feature model class, hyla|rff|none')
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
    parser.add_argument('-metric', choices=['acc', 'f1', 'mr'], default='acc', help='what metrics to report')    
    parser.add_argument('-lambda_scale', type=float, default=0.07, help='scale of lambdas when generating HyLa features')
    parser.add_argument('-inductive', action='store_true', default=False, help='whether embed in the feature level, otherwise node level')
    opt = parser.parse_args()
    
    ## comment following lines during hyper-parameter tuning
    if opt.inductive:
        with open(f'{currentdir}/hyper_parameters_inductive.json',) as f:
            hyper_parameters = json.load(f)[opt.dataset][opt.manifold]
    else:
        with open(f'{currentdir}/hyper_parameters_transductive.json',) as f:
            hyper_parameters = json.load(f)[opt.dataset]
        opt.order = hyper_parameters['order']
    opt.he_dim = hyper_parameters['he_dim']
    opt.hyla_dim = hyper_parameters['hyla_dim']
    opt.lambda_scale = hyper_parameters['lambda_scale']
    opt.lr_e = hyper_parameters['lr_e']
    opt.lr_c = hyper_parameters['lr_c']
    opt.epochs = hyper_parameters['epochs']
    
    opt.metric = 'mr' if opt.dataset =='mr' else 'acc'
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
    
    sp_adj, index_dict, label_dict = load_corpus(f'{currentdir}/datasets/processed_data', opt.dataset, inductive=opt.inductive)
    if opt.progress:
        log.info(f'size of the loaded feature matrix: {sp_adj.shape}')
#     if sp_adj.shape[0]>200000000:#pick a better threshold for batch training
#         log.info('# of training samples >2000, switch to batch training due to memory')
#         opt.batchtraining = True
    
    ## set up label dict
    for k, v in label_dict.items():
        if opt.dataset == "mr":
            label_dict[k] = torch.Tensor(v)#.to(args.device)
        else:
            label_dict[k] = torch.LongTensor(v)#.to(args.device)
    if opt.dataset == "mr": nclass = 1
    else: nclass = label_dict["train"].max().item()+1
    
    if opt.inductive:
        if opt.progress:
            log.info('HyLa at feature level for inductive experiments, data processing')
        adj_dense = sparse_to_torch_dense(sp_adj, device='cpu').double()
        feat_dict = {'train':adj_dense[index_dict['train'], :], 'val':adj_dense[index_dict['val'], :], 'test':adj_dense[index_dict['test'], :]}
    else:
        if opt.progress:
            log.info('HyLa at feature level for transductive experiments, data processing')
        if opt.dataset=='20ng':
            with open(f"{currentdir}/datasets/processed_data/20ng.pkl", "rb") as prep:
                feat_dict =  pkl.load(prep)
            feat_dict = {'train':feat_dict['train'].double(), 'val':feat_dict['val'].double(), 'test':feat_dict['test'].double()}
        else:
            adj_dense = sparse_to_torch_dense(sp_adj, device='cpu')
            if opt.order==1:#accuracy doesn't change, explore why
                feat_dict = {'train':adj_dense[index_dict['train'], :].double(), 'val':adj_dense[index_dict['val'], :].double(), 'test':adj_dense[index_dict['test'], :].double()}
            else:
                adj = sparse_to_torch_sparse(sp_adj, device='cpu')
                feat_dict, precompute_time = sgc_precompute_text(adj, adj_dense, opt.order-1, index_dict)
    feature_dim = feat_dict['train'].size(1)
    if opt.progress:
        log.info(f'dimensionality of original features: {feature_dim}')
    
    # build models and setup optimizers
    model_f = build_model(opt, feature_dim).to(opt.device) #useless for none model
    if opt.manifold in ['euclidean', 'none']:
        optimizer_f = torch.optim.Adam(model_f.parameters(), lr=opt.lr_e)# weight_decay=1.3e-5
#         optimizer_f = torch.optim.SGD(model_f.parameters(), lr=opt.lr_e)# weight_decay=1.3e-5
    elif opt.manifold == 'poincare':
        optimizer_f = RiemannianSGD(model_f.optim_params(), lr=opt.lr_e)
    else:
        raise NotImplemented
    
    model_c = get_model("SGC", opt.hyla_dim , nclass).to(opt.device)
    if opt.optim_type == 'sgd':
        optimizer_c = torch.optim.SGD(model_c.parameters(), lr=opt.lr_c)
    elif opt.optim_type == 'adam':
        optimizer_c = torch.optim.Adam(model_c.parameters(), lr=opt.lr_c)# weight_decay=1.3e-5
    else:
        raise NotImplementedError
    
    ckps = None
    if opt.checkpoint:
        # setup checkpoint
        ckp_fm = generate_ckpt(opt, model_f, f'{currentdir}/ckps/fm.pt')
        ckp_cm = generate_ckpt(opt, model_c, f'{currentdir}/ckps/cm.pt')
        ckps = (ckp_fm, ckp_cm)
    train_acc, train_acc_best, val_acc, val_acc_best = train(model_f, model_c, optimizer_f, optimizer_c, 
        feat_dict, label_dict, opt, log, progress=opt.progress, ckps=ckps)
    if opt.checkpoint and ckps is not None:
        state_fm = ckps[0].load()
        state_cm = ckps[1].load()
        model_f.load_state_dict(state_fm['model'])
        model_c.load_state_dict(state_cm['model'])
        if opt.progress:
            log.info(f'early stopping, loading from epoch: {state_fm["epoch"]} with val_acc_best: {state_fm["val_acc_best"]}')
#     test_acc = test_regression(
#         model_f, model_c, feat_dict['test'], label_dict['test'].to(opt.device), metric = opt.metric)
#     test_acc_threshold = {'R8': 93, 'R52': 85, 'ohsumed': 55, 'mr': 67}
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
