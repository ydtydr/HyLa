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
from hype.hyla_utils import sgc_precompute, sgc_precompute_text, sgc_precompute_text_v1, acc_f1, load_data, load_reddit_data, load_corpus, sparse_to_torch_sparse, sparse_to_torch_dense
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

def test_regression(feature_model, class_model, features, test_labels, cmodel='sgc', metric='acc', heat = False, nonemodel=False):
    with torch.no_grad():
        feature_model.eval()
        class_model.eval()
        if not nonemodel:
            if heat:
                HyLa_features = feature_model.heat_kernel()
            else:
                HyLa_features = feature_model()
        if cmodel == 'sgc':
            if nonemodel:
                HyLa_features = features.cuda()
            else:
                HyLa_features = torch.mm(features.to(HyLa_features.device), HyLa_features)
            predictions = class_model(HyLa_features)
        elif cmodel == 'gcn':
            HyLa_features = torch.spmm(features.to(HyLa_features.device), HyLa_features)
            predictions = class_model(HyLa_features, test_index)
        elif cmodel == 'mlp':
            HyLa_features = torch.spmm(features.to(HyLa_features.device), HyLa_features)
            predictions = class_model(HyLa_features, test_index)
        else:
            raise NotImplementedError
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

def train(device,
          model,
          class_model,
          optimizer_emb,
          optimizer_class,
          feat_dict,
          label_dict,
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
        optimizer_class.zero_grad()
        if not opt.nonemodel:
            if opt.heat:
                HyLa_features = model.heat_kernel()
            else:
                HyLa_features = model()
            if opt.cmodel == 'sgc':
                HyLa_features = torch.mm(feat_dict['train'].to(HyLa_features.device), HyLa_features)
        else:
            HyLa_features = feat_dict['train'].to(device)
        predictions = class_model(HyLa_features)
        del HyLa_features
        if opt.dataset == 'mr':
            loss_class = F.binary_cross_entropy(torch.sigmoid(predictions.squeeze()), label_dict['train'].to(device))
        else:
            loss_class = F.cross_entropy(predictions, label_dict['train'].to(device))
        loss_class.backward()
        ########## debug gradients ###########
#         print(model.lt.weight.grad)
# # #         print(class_model.layers[0].W.weight.grad)
#         print(class_model.W.weight.grad)
#         raise
#         print(model.Lambdas)
#         print(model.Lambdas.grad)
        ########## debug gradients ###########
        optimizer_class.step()
        optimizer_emb.step()
        train_acc = test_regression(
            model, class_model, feat_dict['train'], label_dict['train'].to(device),
            cmodel=opt.cmodel, metric = opt.metric, heat=opt.heat, nonemodel=opt.nonemodel)
        val_acc = test_regression(
            model, class_model, feat_dict['val'], label_dict['val'].to(device),
            cmodel=opt.cmodel, metric = opt.metric, heat=opt.heat, nonemodel=opt.nonemodel)
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

def train_batch(device,
          model,
          class_model,
          optimizer_emb,
          optimizer_class,
          feat_dict,
          label_dict,
          opt,
          log,
          progress=False,
          ckps=None,
):
    len_samples = feat_dict['train'].size(0)
    model.train()
    class_model.train()
    val_acc_best = 0.0
    train_acc_best = 0.0
    for epoch in range(opt.epoch_start, opt.epochs):
        t_start = timeit.default_timer()
        optimizer_emb.zero_grad()
        optimizer_class.zero_grad()
        loss_epoch = 0
        for batch_id in range(len_samples//opt.batchsize + 1):
#             print('batch_id', batch_id)
            if opt.heat:
                HyLa_features = model.heat_kernel()
            else:
                HyLa_features = model()
            start_ind = batch_id * opt.batchsize
            end_ind = (batch_id+1) * opt.batchsize
            if end_ind>len_samples: end_ind = len_samples
            assert not start_ind > len_samples
            assert not end_ind > len_samples
            data = feat_dict['train'][start_ind:end_ind, :].to(device)
            print(data)
            raise
            label = label_dict['train'][start_ind:end_ind].to(device)
            if opt.cmodel == 'sgc':
                HyLa_features = torch.mm(data, HyLa_features)
                predictions = class_model(HyLa_features)
            del HyLa_features
            loss_class = F.cross_entropy(predictions, label)
            loss_class.backward()
            loss_epoch += loss_class.item()
            ########## debug gradients ###########
    #         print(model.lt.weight.grad)
    # # #         print(class_model.layers[0].W.weight.grad)
    #         print(class_model.W.weight.grad)
    #         raise
    #         print(model.Lambdas)
    #         print(model.Lambdas.grad)
            ########## debug gradients ###########
            optimizer_class.step()
            optimizer_emb.step()
        train_acc = test_regression(
            model, class_model, feat_dict['train'], label_dict['train'].to(device),
            cmodel=opt.cmodel, metric = opt.metric, heat=opt.heat)
        val_acc = test_regression(
            model, class_model, feat_dict['val'], label_dict['val'].to(device),
            cmodel=opt.cmodel, metric = opt.metric, heat=opt.heat)
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
                f'"loss_c": {loss_epoch/opt.batchsize:.4f}, '
                '}'
            )
                
        gc.collect()
        torch.cuda.empty_cache()
    return train_acc, train_acc_best, val_acc, val_acc_best


def main():
    parser = argparse.ArgumentParser(description='Train Laplacian Net')
    parser.add_argument('-checkpoint', action='store_true', default=False)
    parser.add_argument('-task', type=str, required=True,
                        help='task [nc|text]')
    parser.add_argument('-dataset', type=str, required=True,
                        help='Dataset identifier [cora|disease_nc|pubmed|citeseer|reddit|20ng|R8|R52|ohsumed|MR]')
    parser.add_argument('-batchsize', type=int, default=256,
                        help='batchsize of the training if batch training')
    parser.add_argument('-batchtraining', action='store_true', default=False, help='whether to try in batches due to memory')
    parser.add_argument('-dim', type=int, default=2,
                        help='Hyperbolic Embedding dimension')
    parser.add_argument('-HyLa_fdim', type=int, default=100,
                        help='Laplacian feature dimension')
    parser.add_argument('-order', type=int, default=5,
                        help='order of adjaceny matrix')
    parser.add_argument('-manifold', type=str, default='poincare',
                        choices=['poincare', 'euclidean', 'none'])
    parser.add_argument('-model', type=str, default='laplaNN',
                        choices=['laplaNN', 'EuclaplaNN', 'none'], help='model class')
    parser.add_argument('-cmodel', type=str, default='sgc',
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
    parser.add_argument('-heat', action='store_true', default=False,
                        help='whether to use heat kernel')
    parser.add_argument('-lre_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-optim_type', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('-metric', choices=['acc', 'f1', 'mr'], default='acc', help='what metrics to report')    
    parser.add_argument('-scale', type=float, default=0.07, help='scale of lambdas when generating Laplacian features')
    parser.add_argument('-nonemodel', action='store_true', default=False, help='remove feature model.')
    parser.add_argument('-use_feats', action='store_true', default=False, help='whether embed in the feature level, otherwise node level')
    opt = parser.parse_args()
    opt.epoch_start = 0
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    opt.split_seed = opt.seed
    opt.progress = not opt.quiet
#     opt.batchtraining = False

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger('HyLa')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    # set default tensor type
    torch.set_default_tensor_type('torch.DoubleTensor')
    # set device
    device = torch.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    
    sp_adj, index_dict, label_dict = load_corpus(opt.dataset, feature=opt.use_feats)
    if opt.progress:
        log.info(f'size of the loaded feature matrix: {sp_adj.shape}')
#     if sp_adj.shape[0]>200000000:#pick a better threshold for batch training
#         log.info('# of training samples >2000, switch to batch training due to memory')
#         opt.batchtraining = True
    for k, v in label_dict.items():
        if opt.dataset == "mr":
            label_dict[k] = torch.Tensor(v)#.to(args.device)
        else:
            label_dict[k] = torch.LongTensor(v)#.to(args.device)
    if opt.dataset == "mr": nclass = 1
    else: nclass = label_dict["train"].max().item()+1
    
    if opt.cmodel == 'sgc':
        if opt.progress:
            log.info('precomputing features if adjacent matrix exists')
        if opt.use_feats:
            if opt.progress:
                log.info('feature level data processing')
            adj_dense = sparse_to_torch_dense(sp_adj, device='cpu').double()
            feat_dict = {'train':adj_dense[index_dict['train'], :], 'val':adj_dense[index_dict['val'], :], 'test':adj_dense[index_dict['test'], :]}
        else:
            if opt.progress:
                log.info('all level data processing')
            if opt.dataset=='20ng':
                with open("./text/preprocessed/20ng.pkl", "rb") as prep:
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
        if opt.nonemodel:
            indim = feature_dim
        else:
            indim = feature_dim if opt.heat else 1*opt.HyLa_fdim 
        # build models and setup optimizers
        model = build_model(opt, feature_dim).to(device) #useless for none model
        class_model = get_model("SGC", indim, nclass).to(device)
    else:
        raise NotImplementedError
    
    if opt.progress:
        # Build config string for log
        log.info(f'json_conf: {json.dumps(vars(opt))}')
        
    if opt.manifold in ['euclidean', 'none']:
#         optimizer_emb = torch.optim.Adam(model.parameters(), lr=opt.lr_e)# weight_decay=1.3e-5
        optimizer_emb = torch.optim.SGD(model.parameters(), lr=opt.lr_e)# weight_decay=1.3e-5
    elif opt.manifold == 'poincare':
        optimizer_emb = RiemannianSGD(model.optim_params(), lr=opt.lr_e)
    else:
        raise NotImplemented
    
    if opt.optim_type == 'sgd':
        optimizer_class = torch.optim.SGD(class_model.parameters(), lr=opt.lr_c)
    elif opt.optim_type == 'adam':
        optimizer_class = torch.optim.Adam(class_model.parameters(), lr=opt.lr_c)# weight_decay=1.3e-5
    elif opt.optim_type == 'lbfgs':
        optimizer_class = torch.optim.LBFGS(class_model.parameters(), lr=opt.lr_c)
    else:
        raise NotImplementedError
    
    ckps = None
    if opt.checkpoint:
        # setup checkpoint
        ckp_fm = generate_ckpt(opt, model, './text/fm.pt')
        cls_m = generate_ckpt(opt, model, './text/clm.pt')
        ckps = (ckp_fm, cls_m)
    if opt.batchtraining:
        train_acc, train_acc_best, val_acc, val_acc_best = train_batch(
            device, model, class_model, optimizer_emb, optimizer_class, 
            feat_dict, label_dict, opt, log, progress=opt.progress, ckps=ckps)
    else:
        train_acc, train_acc_best, val_acc, val_acc_best = train(
            device, model, class_model, optimizer_emb, optimizer_class, 
            feat_dict, label_dict, opt, log, progress=opt.progress, ckps=ckps)
    if opt.checkpoint and ckps is not None:
        state_fm = ckps[0].load()
        state_clm = ckps[1].load()
        model.load_state_dict(state_fm['model'])
        class_model.load_state_dict(state_clm['model'])
        if opt.progress:
            log.info(f'early stopping? loading from epoch: {state_fm["epoch"]} with val_acc_best: {state_fm["val_acc_best"]}')
    test_acc = test_regression(
        model, class_model, feat_dict['test'], label_dict['test'].to(device),
        cmodel=opt.cmodel, metric = opt.metric, heat=opt.heat, nonemodel=opt.nonemodel)
    log.info(
            f'"|| last train_acc": {train_acc*100.0:.2f}%, '
            f'"|| best train_acc": {train_acc_best*100.0:.2f}%, '
            f'"|| last val_acc": {val_acc*100.0:.2f}%, '
            f'"|| best val_acc": {val_acc_best*100.0:.2f}%, '
            f'"|| test_acc": {test_acc*100.0:.2f}%.'
        )

if __name__ == '__main__':
    main()
