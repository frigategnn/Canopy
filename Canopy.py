r"""This is the first code to compute tree canonical representations with
features which will then be used for LSH based distance computation.
# use ~/home2/temp/env
"""

from functools import cmp_to_key
from typing import Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import igraph as ig
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import degree


import torch
import argparse
from classical_ml_methods import run_classical_ml
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Graph tree classical ml training")
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="seed")

    args = parser.parse_args()
    return args


@dataclass
class qitem:
    v: Union[str, int]
    new_name: int
    hop: int


def khoptree(
    G: ig.Graph,
    v: int,
    k: int,
    avg_degree: int,
    seed: int = 0,
) -> ig.Graph:
    r"""Computes k-hop tree of graph"""
    tree = ig.Graph(directed=True)
    # BFS
    queue = [qitem(v, 0, 0)]
    ctr = 1
    tree.add_vertex(0)
    tree.vs[0]["node"] = "n0"
    tree.vs[0]["og_name"] = G.vs[v]["node"]
    tree.vs[0]["index"] = 0
    # print(f"added {tree.vs[0]['og_name']}")
    rng = np.random.RandomState(seed=seed)
    while queue:
        topitem = queue.pop(0)
        current_node = topitem.v
        current_hop = topitem.hop
        current_node_new_name = topitem.new_name
        if current_hop >= k:
            continue
        if current_node != "dummy":
            current_node_neigh = G.neighbors(current_node)
            if len(current_node_neigh) > avg_degree:
                current_node_neigh = np.asarray(current_node_neigh)
                scores = np.asarray([G.vs[k]["score"] for k in current_node_neigh])
                sorting_index = np.argsort(scores)[::-1]
                avg_degree_by_2 = avg_degree // 2
                topk = current_node_neigh[sorting_index[:avg_degree_by_2]]
                rest = current_node_neigh[sorting_index[avg_degree_by_2:]]
                rest_scores = scores[sorting_index[avg_degree_by_2:]]
                rest_probs = rest_scores / rest_scores.sum()
                rest_sample = rng.choice(
                    rest, replace=False, p=rest_probs, size=avg_degree - avg_degree_by_2
                )
                current_node_neigh = topk.tolist() + rest_sample.tolist()
        else:
            current_node_neigh = []
        if len(current_node_neigh) < avg_degree:
            underflow = avg_degree - len(current_node_neigh)
            current_node_neigh.extend(["dummy"] * underflow)
        for nbr in current_node_neigh:
            tree.add_vertex(ctr)
            tree.add_edge(current_node_new_name, ctr)
            tree.vs[ctr]["node"] = f"n{ctr}"
            if nbr != "dummy":
                tree.vs[ctr]["og_name"] = G.vs[nbr]["node"]
            else:
                tree.vs[ctr]["og_name"] = "dummy"
            # print(
            #     f"{' ' * 3 * (current_hop)}{'-' * 3} added {tree.vs[ctr]['og_name']} with parent {tree.vs[current_node_new_name]['og_name']}"
            # )
            queue.append(
                qitem(
                    nbr,
                    ctr,
                    current_hop + 1,
                )
            )
            ctr += 1

    return tree


def build_augmented_feature_for_tree_from_BFCF(BFCF, data, avg_degree):
    dollar_split_segments = [k.strip().strip("#").split() for k in BFCF.split("$")]
    # first segment should just be root
    # but at all other levels, we need ${avg_degree} number of nodes
    for segment in dollar_split_segments[1:]:
        len_segment = len(segment)
        if len_segment < avg_degree:
            len_left = avg_degree - len_segment
            segment.extend(["dummy"] * len_left)
    feature_dim = data.shape[1]
    features_to_cat = []
    last_idx_dollar_ss = len(dollar_split_segments) - 1
    for idx, segment in enumerate(dollar_split_segments):
        for node in segment:
            if node == "dummy":
                # handle_dummy()
                dummy_features = torch.ones(feature_dim) * -1
                features_to_cat.append(dummy_features)
                continue
            # handle_normal()
            # node is of the form n1 n2 n100 n200 etc.
            nid = int(node[1:])
            node_features = data[nid]
            features_to_cat.append(node_features)
        # handle_dollar()
        if idx != last_idx_dollar_ss:
            dollar_features = torch.ones(1) * -1
            features_to_cat.append(dollar_features)
    features = torch.cat(features_to_cat)
    return features


def normalize_tokens(s, markers=({"$", "#"})):
    """
    Split a level-order string into tokens.
    Handles cases like "n1$" -> ["n1", "$"] and "n1" -> ["n1"].
    """
    toks = []
    for part in s.split():
        if not part:
            continue
        # if the last char is a marker, split it off (handles single trailing marker)
        if part[-1] in markers:
            base, mark = part[:-1], part[-1]
            if base:
                toks.append(base)
            toks.append(mark)
        else:
            toks.append(part)
    return toks


def compare_vectors(v1, v2):
    """
    Lexicographic compare of two numeric sequences.
    Returns -1 if v1 < v2, 1 if v1 > v2, 0 if equal.
    """
    # allow lists/tuples/numpy arrays
    la, lb = len(v1), len(v2)
    for x, y in zip(v1, v2):
        if x < y:
            return -1
        if x > y:
            return 1
    if la < lb:
        return -1
    if la > lb:
        return 1
    return 0


def level_order_comparator(a, b, data):
    """
    Comparator for two level-order traversal strings a and b.
    `data` is a dict: node_id -> numeric vector (list/tuple/np.array).
    Ordering rule (small -> large): nodes < '$' < '#'.
    If both tokens are node ids, compare data[node_id] lexicographically.
    If a node id is missing in `data`, fall back to string compare of ids.
    """
    MARK_RANK = {"dummy": 1, "$": 2, "#": 3}  # higher rank => larger token
    ta = normalize_tokens(a)
    tb = normalize_tokens(b)
    n = max(len(ta), len(tb))
    for i in range(n):
        if i >= len(ta):  # a exhausted -> smaller (prefix rule)
            return -1
        if i >= len(tb):
            return 1
        xa, xb = ta[i], tb[i]
        if xa == xb:
            continue
        a_is_mark = xa in MARK_RANK
        b_is_mark = xb in MARK_RANK
        # If any is a marker: markers are larger than node ids; compare marker ranks if both markers
        if a_is_mark or b_is_mark:
            if a_is_mark and b_is_mark:
                return -1 if MARK_RANK[xa] < MARK_RANK[xb] else 1
            # one is marker, other is node -> marker is larger
            return 1 if a_is_mark else -1
        # both are node ids -> compare feature vectors
        node_id_a = int(xa[1:])
        node_id_b = int(xb[1:])
        va = data[node_id_a]
        vb = data[node_id_b]
        if va is None or vb is None:
            # fallback: string compare (stable deterministic)
            return -1 if xa < xb else 1
        vec_cmp = compare_vectors(va, vb)
        if vec_cmp != 0:
            return vec_cmp
        # else equal vectors -> continue to next token
    return 0


def breadth_first_canonical_form(tree: ig.Graph, *, data, root=None):
    def level_order_traversal(BFCF_order):
        Q = [BFCF_order]
        levels = [0]
        parents = [-1]
        prev_parent = -1
        s = ""
        while Q:
            front = Q.pop(0)
            level = levels.pop(0)
            parent = parents.pop(0)
            if parent != prev_parent:
                s += "$"
            prev_parent = parent
            if isinstance(front, list):
                root = front[0]
                children = front[1:]
                s += f" {tree.vs[root]['og_name']}"
                [Q.append(c) for c in children]
                [levels.append(level + 1) for c in children]
                [parents.append(root) for c in children]
            else:
                s += f" {tree.vs[front]['og_name']}"
        return s + "#"

    def helper(root):
        if tree.vs[root].outdegree() == 0:
            return root
        children = [helper(root=c) for c in tree.neighbors(root, mode="out")]
        # children_level_order = list(
        #     map(
        #         lambda x: level_order_traversal(x),
        #         children,
        #     )
        # )
        # print(
        #     children_level_order,
        #     list(
        #         map(
        #             lambda x: level_order_traversal(x),
        #             sorted(
        #                 children,
        #                 key=cmp_to_key(
        #                     lambda x, y: level_order_comparator(
        #                         level_order_traversal(x), level_order_traversal(y), data
        #                     )
        #                 ),
        #             ),
        #         )
        #     ),
        # )
        sorted_children = sorted(
            children,
            key=cmp_to_key(
                lambda x, y: level_order_comparator(
                    level_order_traversal(x), level_order_traversal(y), data
                )
            ),
        )
        return [root] + sorted_children

    if root is None:
        (root,) = np.where([v.indegree() == 0 for v in tree.vs])
        root = root.item()
    BFCF_order = helper(root)
    return level_order_traversal(BFCF_order)


def main():
    r"""The main code. For Planetoid dataset currently. Computes all
    featurized labels.
    Warning: no tests have been performed to check correctness of the
    featurization.
    """
    args = parse_args()

    # dataset = Planetoid(root="data", name="cora")
    args.dataset_name = args.dataset_path.stem
    dataset_dict = torch.load(args.dataset_path)
    graph = dataset_dict["graph"]
    data = Data(
        x=graph["node_feat"],
        edge_index=graph["edge_index"],
        y=dataset_dict["label"].squeeze(),
    )
    print("DEBUG LOG::", len(set(y.item() for y in data.y)))

    # Override masks to 60% train, 20% val, 20% test
    n_vertices = data.x.shape[0]
    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_vertices, generator=gen)
    train_end = int(0.6 * n_vertices)
    val_end = int(0.8 * n_vertices)
    train_mask = torch.zeros(n_vertices, dtype=torch.bool)
    val_mask = torch.zeros(n_vertices, dtype=torch.bool)
    test_mask = torch.zeros(n_vertices, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    degrees = degree(data.edge_index[1], num_nodes=data.x.shape[0])
    norms = torch.norm(data.x, dim=1)
    alpha, beta = 1, 1
    scores = alpha * degrees + beta * norms
    scores = scores / scores.sum()

    n_vertices = data.x.shape[0]
    row, col = data.edge_index
    row, col = row.numpy(), col.numpy()
    val = np.ones_like(row)
    adj = sp.coo_matrix((val, (row, col)), shape=(n_vertices, n_vertices))
    adj = adj - sp.eye(n_vertices)
    adj = adj.astype(int)
    G = ig.Graph.Adjacency(adj, mode="undirected")
    G.vs["node"] = [f"n{i}" for i in range(len(G.vs))]
    G.vs["score"] = [score.item() for score in scores]
    hop = 2

    median_degree = int(np.median(G.degree()))
    avg_degree = min(median_degree, 7)
    print(f"DEBUG LOG:: {median_degree = }")

    train_trees = [
        khoptree(G, node, hop, avg_degree) for node in range(data.x.shape[0])
    ]
    features = data.x.numpy()

    pca_components = 50
    pca = PCA(n_components=pca_components, random_state=42)
    features_pca = pca.fit_transform(features)
    features_pca = torch.tensor(features_pca)
    features_pca = torch.cat(
        [features_pca, torch.tensor(G.degree()).reshape(-1, 1)],
        dim=1,
    )
    BFCFs = [
        breadth_first_canonical_form(tree, data=features_pca) for tree in train_trees
    ]

    augmented_feature_list = []
    for BFCF in BFCFs:
        vec = build_augmented_feature_for_tree_from_BFCF(
            BFCF,
            features_pca,
            avg_degree=avg_degree,
        )
        augmented_feature_list.append(vec.numpy())
    print(
        f"DEBUG LOG:: Unique shapes found in augmented features: {np.unique([k.shape[0] for k in augmented_feature_list])}"
    )
    augmented_features = np.vstack(augmented_feature_list)
    print("DEBUG LOG:: Augmented feature matrix shape:", augmented_features.shape)

    result_root = Path("BFCF_results")
    result_root.mkdir(exist_ok=True)
    results_df = run_classical_ml(
        augmented_features,
        data,
        output_csv_path=result_root
        / f"{args.dataset_name}_classical_ml_results_with_hetero_hops.csv",
    )
    print(results_df)


if __name__ == "__main__":
    main()
