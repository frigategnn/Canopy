from pathlib import Path
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from eval import *
from parse import parse_method, parser_add_main_args


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ### Parse args ###
    parser = argparse.ArgumentParser(
        description="Training Pipeline for Node Classification"
    )
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    ### Load and preprocess data ###
    dataset = load_dataset(args.data_dir, args.dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    if args.rand_split:
        split_idx_lst = [
            dataset.get_idx_split(
                train_prop=args.train_prop, valid_prop=args.valid_prop
            )
            for _ in range(args.runs)
        ]
    elif args.rand_split_class:
        split_idx_lst = [
            class_rand_splits(
                dataset.label, args.label_num_per_class, args.valid_num, args.test_num
            )
        ]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    dataset.label = dataset.label.to(device)

    ### Basic information of datasets ###
    n = dataset.graph["num_nodes"]
    e = dataset.graph["edge_index"].shape[1]
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph["node_feat"].shape[1]
    dataset.graph["num_nodes"] = n
    dataset.graph["num_edges"] = e
    dataset.graph["num_classes"] = c
    dataset.graph["feature_dim"] = d
    dataset.graph["edge_feature_dim"] = (
        0 if dataset.graph["edge_feat"] is None else dataset.graph["edge_feat"].shape[1]
    )

    print(
        f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}"
    )

    dataset.graph["edge_index"] = to_undirected(dataset.graph["edge_index"])
    dataset.graph["edge_index"], _ = remove_self_loops(dataset.graph["edge_index"])
    dataset.graph["edge_index"], _ = add_self_loops(
        dataset.graph["edge_index"], num_nodes=n
    )

    dataset.graph["edge_index"], dataset.graph["node_feat"] = (
        dataset.graph["edge_index"].to(device),
        dataset.graph["node_feat"].to(device),
    )
    # =============================================================
    # INFO: Mridul
    # =============================================================
    data_root = Path("./save_dataset_with_splits")
    data_root.mkdir(exist_ok=True)
    data_subdir = data_root / args.dataset
    data_subdir.mkdir(exist_ok=True)
    data_dump = vars(dataset)
    data_dump["splits_to_use"] = split_idx_lst
    data_dump["note"] = (
        "In some datasets, there's separate train_idx, valid_idx, etc. Don't use them, use `splits_to_use`."
    )

    def traverse_tree_and_move_to_cpu(json_tree):
        if isinstance(json_tree, dict):
            subtree = {}
            for key, value in json_tree.items():
                subtree[key] = traverse_tree_and_move_to_cpu(value)
            return subtree
        elif isinstance(json_tree, torch.Tensor):
            return json_tree.to("cpu")
        else:
            return json_tree

    data_dump = traverse_tree_and_move_to_cpu(data_dump)
    data_dump_file_name = data_subdir / f"{args.dataset}.pth"
    torch.save(data_dump, data_dump_file_name)
    print(f"Successfully saved {data_dump_file_name}.")


if __name__ == "__main__":
    main()
