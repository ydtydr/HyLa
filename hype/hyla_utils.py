#!/usr/bin/env python3
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
import os
import pickle as pkl
import sys
import networkx as nx
import torch.nn.functional as F
import json
from networkx.readwrite import json_graph
import pdb
from scipy.sparse.linalg.eigen.arpack import eigsh
import re
from time import perf_counter
import tabulate
sys.setrecursionlimit(99999)


def sample_boundary(n_Bs, d, cls):
    if cls =='RandomUniform' or d>2:
        pre_b = torch.randn(n_Bs, d)
        b = pre_b/torch.norm(pre_b,dim=-1,keepdim=True)
    elif cls == 'FixedUniform':
        theta = torch.arange(0,2 * np.pi, 2*np.pi/n_Bs)
        b = torch.stack([torch.cos(theta), torch.sin(theta)],1)
    elif cls == 'RandomDisk':
        theta = 2 * np.pi * torch.rand(n_Bs)
        b = torch.stack([torch.cos(theta), torch.sin(theta)],1)
    else:
        raise NotImplementedError
    return b

def PoissonKernel(X, b):
    X = X.view(X.size(0), 1, X.size(-1))
    return (1 - torch.norm(X, 2, dim=-1)**2)/(torch.norm(X-b, 2, dim=-1)**2)
#     return (1 - torch.sum(X * X, dim=-1))/torch.sum((X-b)**2,dim=-1)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float64)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.DoubleTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def add_self_loop(adj):
    return adj + sp.eye(adj.shape[0])

def sgc_precompute(adj, features, degree):
    nonzero_perc = []
#     assert degree > 0, 'invalid degree as 0'
    if degree==0:
        number_nonzero = (features != 0).sum().item()
        percentage = number_nonzero*1.0/features.size(0)/features.size(1)*100.0
        nonzero_perc.append("%.2f" % percentage)
        print('input order 0, return raw feature')
        return features, nonzero_perc
    for i in range(degree):
        features = torch.spmm(adj, features)
        number_nonzero = (features != 0).sum().item()
        percentage = number_nonzero*1.0/features.size(0)/features.size(1)*100.0
        nonzero_perc.append("%.2f" % percentage)
    return features, nonzero_perc

def acc_f1(output, labels, average='micro'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1

def measure_tensor_size(a):
    # return # MB
    return a.element_size() * a.nelement() * 0.000001




# ###################################################
# data loading

def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
#     data['adj_train_norm'], data['features'] = process(
#             data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
#     )
#     if args.dataset == 'airport':
#         data['features'] = augment(data['adj_train'], data['features'])
    adj_n = aug_normalized_adjacency(data['adj_train'])
    data['adj_train'] = sparse_mx_to_torch_sparse_tensor(adj_n)
#     data['adj_train'] = sparse_mx_to_torch_sparse_tensor(add_self_loop(data['adj_train']))
#     if sp.isspmatrix(data['features']):
#         data['features'] = sparse_mx_to_torch_sparse_tensor(data['features'])
#         features = np.array(data['features'].todense())
#     features = normalize(features)
#     data['features'] = torch.Tensor(features)

    data['features'] = sparse_mx_to_torch_sparse_tensor(data['features'])
    return data

# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T # remove maybe?
    train_adj = adj[train_index, :][:, train_index]
    features = torch.tensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)    
    
    adj = aug_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)#.float()
    train_adj = aug_normalized_adjacency(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj)#.float()
    labels = torch.LongTensor(labels)
    
    data = {'adj_all': adj, 'adj_train': train_adj, 'features': features, 'labels': labels, 'idx_train': train_index, 'idx_val': val_index, 'idx_test': test_index}
    
    return data


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
#     graph = pkl.load(open(datapath, 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), sp.csr_matrix(features), labels
    else:
        return sp.csr_matrix(adj), sp.csr_matrix(features)
    
# ############### Loading ppi ####################################    
# adapted from PetarV/GAT
def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        #for v in range(nb_nodes):
        for v in adj[u,:].nonzero()[1]:
            #if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)

def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret

def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] != mapping[j]:
              #  if adj[i,j] == 1:
                 return False
    return True

def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits={}
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i]==0 or mapping[j]==0:
                dict_splits[0]=None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:

                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]['val']:
                            dict_splits[mapping[i]] = 'val'

                        elif ds_label[i]['test']:
                            dict_splits[mapping[i]]='test'

                        else:
                            dict_splits[mapping[i]] = 'train'

                    else:
                        if ds_label[i]['test']:
                            ind_label='test'
                        elif ds_label[i]['val']:
                            ind_label='val'
                        else:
                            ind_label='train'
                        if dict_splits[mapping[i]]!= ind_label:
                            print ('inconsistent labels within a graph exiting!!!')
                            return None
                else:
                    print ('label of both nodes different, exiting!!')
                    return None
    return dict_splits

def load_ppi(data_path):

    print ('Loading G...')
    with open(data_path + 'ppi-G.json') as jsonfile:
        g_data = json.load(jsonfile)
    # print (len(g_data))
    G = json_graph.node_link_graph(g_data)

    #Extracting adjacency matrix
    adj=nx.adjacency_matrix(G)

    prev_key=''
    for key, value in g_data.items():
        if prev_key!=key:
            # print (key)
            prev_key=key

    # print ('Loading id_map...')
    with open(data_path + 'ppi-id_map.json') as jsonfile:
        id_map = json.load(jsonfile)
    # print (len(id_map))

    id_map = {int(k):int(v) for k,v in id_map.items()}
    for key, value in id_map.items():
        id_map[key]=[value]
    # print (len(id_map))

    print ('Loading features...')
    features_=np.load(data_path + 'ppi-feats.npy')
    # print (features_.shape)

    #standarizing features
    from sklearn.preprocessing import StandardScaler

    train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
    train_feats = features_[train_ids[:,0]]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    features_ = scaler.transform(features_)

    features = sp.csr_matrix(features_).tolil()


    print ('Loading class_map...')
    class_map = {}
    with open(data_path + 'ppi-class_map.json') as jsonfile:
        class_map = json.load(jsonfile)
    # print (len(class_map))
    
    #pdb.set_trace()
    #Split graph into sub-graphs
    # print ('Splitting graph...')
    splits=dfs_split(adj)

    #Rearrange sub-graph index and append sub-graphs with 1 or 2 nodes to bigger sub-graphs
    # print ('Re-arranging sub-graph IDs...')
    list_splits=splits.tolist()
    group_inc=1

    for i in range(np.max(list_splits)+1):
        if list_splits.count(i)>=3:
            splits[np.array(list_splits) == i] =group_inc
            group_inc+=1
        else:
            #splits[np.array(list_splits) == i] = 0
            ind_nodes=np.argwhere(np.array(list_splits) == i)
            ind_nodes=ind_nodes[:,0].tolist()
            split=None
            
            for ind_node in ind_nodes:
                if g_data['nodes'][ind_node]['val']:
                    if split is None or split=='val':
                        splits[np.array(list_splits) == i] = 21
                        split='val'
                    else:
                        raise ValueError('new node is VAL but previously was {}'.format(split))
                elif g_data['nodes'][ind_node]['test']:
                    if split is None or split=='test':
                        splits[np.array(list_splits) == i] = 23
                        split='test'
                    else:
                        raise ValueError('new node is TEST but previously was {}'.format(split))
                else:
                    if split is None or split == 'train':
                        splits[np.array(list_splits) == i] = 1
                        split='train'
                    else:
                        pdb.set_trace()
                        raise ValueError('new node is TRAIN but previously was {}'.format(split))

    #counting number of nodes per sub-graph
    list_splits=splits.tolist()
    nodes_per_graph=[]
    for i in range(1,np.max(list_splits) + 1):
        nodes_per_graph.append(list_splits.count(i))

    #Splitting adj matrix into sub-graphs
    subgraph_nodes=np.max(nodes_per_graph)
    adj_sub=np.empty((len(nodes_per_graph), subgraph_nodes, subgraph_nodes))
    feat_sub = np.empty((len(nodes_per_graph), subgraph_nodes, features.shape[1]))
    labels_sub = np.empty((len(nodes_per_graph), subgraph_nodes, 121))

    for i in range(1, np.max(list_splits) + 1):
        #Creating same size sub-graphs
        indexes = np.where(splits == i)[0]
        subgraph_=adj[indexes,:][:,indexes]

        if subgraph_.shape[0]<subgraph_nodes or subgraph_.shape[1]<subgraph_nodes:
            subgraph=np.identity(subgraph_nodes)
            feats=np.zeros([subgraph_nodes, features.shape[1]])
            labels=np.zeros([subgraph_nodes,121])
            #adj
            subgraph = sp.csr_matrix(subgraph).tolil()
            subgraph[0:subgraph_.shape[0],0:subgraph_.shape[1]]=subgraph_
            adj_sub[i-1,:,:]=subgraph.todense()
            #feats
            feats[0:len(indexes)]=features[indexes,:].todense()
            feat_sub[i-1,:,:]=feats
            #labels
            for j,node in enumerate(indexes):
                labels[j,:]=np.array(class_map[str(node)])
            labels[indexes.shape[0]:subgraph_nodes,:]=np.zeros([121])
            labels_sub[i - 1, :, :] = labels

        else:
            adj_sub[i - 1, :, :] = subgraph_.todense()
            feat_sub[i - 1, :, :]=features[indexes,:].todense()
            for j,node in enumerate(indexes):
                labels[j,:]=np.array(class_map[str(node)])
            labels_sub[i-1, :, :] = labels

    # Get relation between id sub-graph and tran,val or test set
    dict_splits = find_split(adj, splits, g_data['nodes'])

    # Testing if sub graphs are isolated
    # print ('Are sub-graphs isolated?')
    # print (test(adj, splits))

    #Splitting tensors into train,val and test
    train_split=[]
    val_split=[]
    test_split=[]
    for key, value in dict_splits.items():
        if dict_splits[key]=='train':
            train_split.append(int(key)-1)
        elif dict_splits[key] == 'val':
            val_split.append(int(key)-1)
        elif dict_splits[key] == 'test':
            test_split.append(int(key)-1)

    train_adj=adj_sub[train_split,:,:]
    val_adj=adj_sub[val_split,:,:]
    test_adj=adj_sub[test_split,:,:]

    train_feat=feat_sub[train_split,:,:]
    val_feat = feat_sub[val_split, :, :]
    test_feat = feat_sub[test_split, :, :]

    train_labels = labels_sub[train_split, :, :]
    val_labels = labels_sub[val_split, :, :]
    test_labels = labels_sub[test_split, :, :]

    train_nodes=np.array(nodes_per_graph[train_split[0]:train_split[-1]+1])
    val_nodes = np.array(nodes_per_graph[val_split[0]:val_split[-1]+1])
    test_nodes = np.array(nodes_per_graph[test_split[0]:test_split[-1]+1])


    #Masks with ones

    tr_msk = np.zeros((len(nodes_per_graph[train_split[0]:train_split[-1]+1]), subgraph_nodes))
    vl_msk = np.zeros((len(nodes_per_graph[val_split[0]:val_split[-1] + 1]), subgraph_nodes))
    ts_msk = np.zeros((len(nodes_per_graph[test_split[0]:test_split[-1]+1]), subgraph_nodes))

    for i in range(len(train_nodes)):
        for j in range(train_nodes[i]):
            tr_msk[i][j] = 1

    for i in range(len(val_nodes)):
        for j in range(val_nodes[i]):
            vl_msk[i][j] = 1

    for i in range(len(test_nodes)):
        for j in range(test_nodes[i]):
            ts_msk[i][j] = 1

    train_adj_list = []
    val_adj_list = []
    test_adj_list = []
    for i in range(train_adj.shape[0]):
        adj = sp.coo_matrix(train_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = aug_normalized_adjacency(adj)
        train_adj_list.append(sparse_mx_to_torch_sparse_tensor(tmp))
    for i in range(val_adj.shape[0]):
        adj = sp.coo_matrix(val_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = aug_normalized_adjacency(adj)
        val_adj_list.append(sparse_mx_to_torch_sparse_tensor(tmp))
        adj = sp.coo_matrix(test_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = aug_normalized_adjacency(adj)
        test_adj_list.append(sparse_mx_to_torch_sparse_tensor(tmp))

    train_feat = torch.tensor(train_feat)
    val_feat = torch.tensor(val_feat)
    test_feat = torch.tensor(test_feat)

    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    test_labels = torch.tensor(test_labels)

    tr_msk = torch.LongTensor(tr_msk)
    vl_msk = torch.LongTensor(vl_msk)
    ts_msk = torch.LongTensor(ts_msk)

    return train_adj_list,val_adj_list,test_adj_list,train_feat,val_feat,test_feat,train_labels,val_labels, test_labels, train_nodes, val_nodes, test_nodes

# ############### Loading for TextHyLa ####################################    
# adapted from Tiiiger/SGC
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_corpus(dataset_str, feature=False):
    """
    Loads input corpus from text/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    index_dict = {}
    label_dict = {}
    phases = ["train", "val", "test"]
    objects = []
    def load_pkl(path):
        with open(path.format(dataset_str, p), 'rb') as f:
            if sys.version_info > (3, 0):
                return pkl.load(f, encoding='latin1')
            else:
                return pkl.load(f)

    for p in phases:
        index_dict[p] = load_pkl("./text/data/ind.{}.{}.x".format(dataset_str, p))
        label_dict[p] = load_pkl("./text/data/ind.{}.{}.y".format(dataset_str, p))

    if feature:
        adj = load_pkl("./text/data/ind.{}.B.adj".format(dataset_str))
        adj = adj.astype(np.float32)
    else:
        adj = load_pkl("./text/data/ind.{}.BCD.adj".format(dataset_str))
        adj = adj.astype(np.float32)
#         print(adj)
#         print(adj.shape)
#         raise
        adj = aug_normalized_adjacency(adj)

    return adj, index_dict, label_dict

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    string = re.sub(r'[?|$|.|!]',r'',string)
    string = re.sub(r'[^a-zA-Z0-9 ]',r'',string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def sparse_to_torch_sparse(sparse_mx, device='cuda'):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    if device == 'cuda':
        indices = indices.cuda()
        values = torch.from_numpy(sparse_mx.data).cuda()
        shape = torch.Size(sparse_mx.shape)
        adj = torch.cuda.sparse.FloatTensor(indices, values, shape)
    elif device == 'cpu':
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        adj = torch.sparse.FloatTensor(indices, values, shape)
    return adj

def sparse_to_torch_dense(sparse, device='cuda'):
    dense = sparse.todense().astype(np.float32)
    torch_dense = torch.from_numpy(dense).to(device=device)
    return torch_dense

def sgc_precompute_text(adj, features, degree, index_dict):
#     assert degree==1, "Only supporting degree 2 now"
    assert degree > 0, 'invalid degree as 0'
    feat_dict = {}
    start = perf_counter()
    train_feats = features[:, index_dict["train"]]#.cuda()
    #     nonzero_perc = []
    for i in range(degree):
        train_feats = torch.spmm(adj, train_feats)
#         number_nonzero = (features != 0).sum().item()
#         percentage = number_nonzero*1.0/features.size(0)/features.size(1)*100.0
#         nonzero_perc.append("%.2f" % percentage)
    train_feats = train_feats.t()
    train_feats_max, _ = train_feats.max(dim=0, keepdim=True)
    train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    train_feats_range = train_feats_max-train_feats_min
    useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze()
    train_feats = train_feats[:, useful_features_dim]
    train_feats_range = train_feats_range[:, useful_features_dim]
    train_feats_min = train_feats_min[:, useful_features_dim]
    train_feats = (train_feats-train_feats_min)/train_feats_range
    feat_dict["train"] = train_feats.double()
    for phase in ["test", "val"]:
        feats = features[:, index_dict[phase]]#.cuda()
        feats = torch.spmm(adj, feats).t()
        feats = feats[:, useful_features_dim]
        feat_dict[phase] = ((feats-train_feats_min)/train_feats_range).cpu().double() # adj is symmetric!
    precompute_time = perf_counter()-start
    return feat_dict, precompute_time

def sgc_precompute_text_v1(adj, features, degree, index_dict):
    assert degree > 0, 'invalid degree as 0'
    feat_dict = {}
    start = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    train_feats = features[index_dict["train"], :].double()
    train_feats_max, _ = train_feats.max(dim=0, keepdim=True)
    train_feats_min, _ = train_feats.min(dim=0, keepdim=True)
    train_feats_range = train_feats_max-train_feats_min
    useful_features_dim = train_feats_range.squeeze().gt(0).nonzero().squeeze()
    train_feats = train_feats[:, useful_features_dim]
    train_feats_range = train_feats_range[:, useful_features_dim]
    train_feats_min = train_feats_min[:, useful_features_dim]
    feat_dict["train"] = (train_feats-train_feats_min)/train_feats_range
    for phase in ["test", "val"]:
        feats = features[index_dict[phase], :].double()
        feats = feats[:, useful_features_dim]
        feat_dict[phase] = ((feats-train_feats_min)/train_feats_range).cpu() # adj is symmetric!
    precompute_time = perf_counter()-start
    return feat_dict, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def print_table(values, columns, epoch):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)



