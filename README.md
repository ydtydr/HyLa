## Laplacian Features for Learning with Hyperbolic Space

#### Authors:
* [Tao Yu](http://www.cs.cornell.edu/~tyu/)
* [Christopher De Sa](http://www.cs.cornell.edu/~cdesa/)

## Introduction
This repo contains official implementation for HyLa (HYperbolic LAplacian features), described in the paper [Laplacian Features for Learning with Hyperbolic Space](https://arxiv.org/abs/2202.06854).

Our HyLa feature, when used together with a simple linear graph model, i.e., SGC, can already achieves impressive performance. For example, on the node classification tasks over standard benchmarks with 16 dimensional hypebolic embeddings (baseline results taken from corresponding papers):

Dataset (Hyperbolicity) | HyLa-SGC | [HGCN](https://arxiv.org/abs/1910.12933) | [SGC](https://arxiv.org/abs/1902.07153) 
:----------------------:|:--------:|:----:|:---:|
Disease (0.0) | 91.3 % | 74.5 % | 69.5%
Airport (1.0) | 95.8% | 90.6 % | 80.6%
Pubmed (3.5) | 81.1 % | 80.3 % | 78.9%
Citeseer (5.0) | 72 % | 64.0 % | 71.9%
Cora (11.0)| 82.6 % | 79.9 % | 81.0%

### Dependencies and Setup
PyTorch>=1.0.0 is required to run the implementation, other dependences are provided in `requirements.txt`, before training, set up environments by simply running 
```
$ conda env create -f environment.yml
$ conda activate hyla
```

### Task and Data
This repo contains implementation for both node classification (`./nc`) and text classification (`./text`) task. For the node classification task, The `./nc/datasets/` folder includes both standard benchmarks (Cora, Citeseer, and Pubmed), social network (Reddit) and hyperbolid datasets (Airport, and Disease). Due to space limit, please download reddit dataset from [FastGCN](https://github.com/matenure/FastGCN/issues/9) and put `reddit_adj.npz`, `reddit.npz` under `./nc/datasets/reddit/`.

Implementation for text classification following ([Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679) and [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)).
Code modified from repo Text-GCN(https://github.com/yao8839836/text_gcn) and SGC(https://github.com/Tiiiger/SGC). Both transductive and inductive settings are considered, where HyLa-SGC outperforms Text-GCN and Text-SGC especially in the inductive setting. We get the text data (R8, R52, Ohsumed, and MR) from [Text-GCN repo](https://github.com/yao8839836/text_gcn) and process with `remove_words.py`, `build_graph.py` following [SGC repo](https://github.com/Tiiiger/SGC) with some slight changes to the code. If you want to redo the processing, see options with `python build_graph.py --help` and `python remove_words.py --help`. Please put processed data into `./text/datasets/processed_data/`.

## Usage
### Node Classification
We tune the hyperparameters with grid search for all datasets at both 16 (`./nc/hyper_parameters_16d.json`) and 50 dimension (`./nc/hyper_parameters_50d.json`). Simply specify the dataset and dimension in `train-nc.sh` and run 
```
$ bash train-nc.sh
```
to get results in the paper. particularly, please remove `-use_feats` option for `airport` dataset and include `-inductive` option for inductive training on `reddit`. For a detailed look of the code, see more options with `python ./nc/HyLa.py --help` and add hyperparameters you want to change in `train-nc.sh`. 

### Text Classification
We tune the hyperparameters with grid search for all datasets at both transductive (`./text/hyper_parameters_transductive.json`) and inductive (`./text/hyper_parameters_inductive.json`) setting. The hyperbolic embedding dimension is fixed at 50 for consistency. Please specify training options (e.g. datasets) in `train-text.sh` and run 
```
$ bash train-text.sh
```
to get results in the paper. See more options with `python ./text/TextHyLa.py --help` and add extra hyperparameters in `train-text.sh` to play with it. 

### Some notes
Early stopping is provided in the code, i.e., resume back to the model with best val accuracy, though it's not used when reporting results. Some of the code was also adapted from [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings) and [hgcn](https://github.com/HazyResearch/hgcn/blob/master/README.md) to set up e.g., baselines,  hyperbolic models and optimizers. 


### Citation
If you find this repo useful, please cite:
```
@article{yu2022hyla,
  title={Laplacian Features for Learning with Hyperbolic Space},
  author={Yu, Tao and De Sa, Christopher},
  journal={arXiv preprint arXiv:2202.06854},
  year={2022}
}
```