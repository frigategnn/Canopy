# Canopy
This repository contains the official code of Canopy. Please see the insturctions below to run the code.
## Datasets
Please download the datasets by running
```
bash download_data.sh
```
## Run canopy
To run canopy, use the following command
```
python run_canopy.py
```

## To run canopy on heterophilous datasets with k-hop modification
```
python Canopy_heterophily.py --dataset_path {dataset_path} --heterophilic_hop_k {k}
```

## To run GNNs
```
python GNN.py
```
## To train using only raw features with classical methods
```
python Only_classical.py
```
## Required libraries
```
numpy
igraph
scipy
torch_geometric
torch
scikit-learn
networkx
pandas
matplotlib
xgboost
ogb
gdown
dgl
```
