import os
import pickle as pkl
import sys
import time
import argparse
import networkx as nx
import numpy as np
from tqdm import tqdm

from hype.hyla_utils import sgc_precompute, acc_f1, load_data, load_reddit_data, load_data_nc

def hyperbolicity_sample(G, num_samples=1000000):#50000  10000000 1000000 5000000
    curr_time = time.time()
    hyps = []
    for i in tqdm(range(num_samples)):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    print('Time for hyp: ', time.time() - curr_time)
    return max(hyps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing hyperbolicity')
    parser.add_argument('-dataset', type=str, required=True,
                        help='Dataset identifier [cora|disease_nc|pubmed|citeseer|reddit]')
    opt = parser.parse_args()
    opt.use_feats = False
    opt.split_seed = 43
    data_path = './nc/' + opt.dataset + '/'
    if opt.dataset in ['cora', 'disease_nc', 'pubmed', 'citeseer', 'airport']:
        data = load_data_nc(opt.dataset, opt.use_feats, data_path, opt.split_seed)
    elif opt.dataset in ['reddit']:
        data = load_reddit_data(data_path)
    else:
        raise NotImplemented
    graph = nx.from_scipy_sparse_matrix(data['adj_train'])
#     print('adj', data['adj_train'])
#     print('graph', graph)
    print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
    hyp = hyperbolicity_sample(graph)
    print('Hyp: ', hyp)

