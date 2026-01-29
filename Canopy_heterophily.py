r"""This is the first code to compute tree canonical representations with
features which will then be used for LSH based distance computation.
# use ~/home2/temp/env
"""

from pathlib import Path
from typing import Union, List, Optional, Dict
from dataclasses import dataclass
import numpy as np
import igraph as ig
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


# from model import GCN
import torch
import random
import argparse
import time
from classical_ml_methods import run_classical_ml
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(description="Coarsened Graph Training")
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--heterophilic_hop_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args


@dataclass
class qitem:
    v: str
    new_name: int
    hop: int
    index: int
    total_child_nodes: int
    parent_index: int


def khoptree(
    G: ig.Graph, v: int, k: int, level_min_max_index: dict, avg_degree: int
) -> ig.Graph:
    r"""Computes k-hop tree of graph"""
    tree = ig.Graph(directed=True)
    # BFS
    queue = [qitem(v, 0, 0, 0, 0, 0)]
    node_tracker = {}  ## after pop we loss nodes from queue this is to keep track of nodes
    ctr = 1
    tree.add_vertex(0)
    tree.vs[0]["node"] = "n0"
    tree.vs[0]["og_name"] = G.vs[v]["node"]
    tree.vs[0]["index"] = 0
    # print(tree.vs[0]["og_name"])
    while queue:
        topitem = queue.pop(0)
        current_node = topitem.v
        current_hop = topitem.hop
        current_node_new_name = topitem.new_name
        key = (
            (str)(current_node) + "_" + (str)(topitem.index)
        )  # + "_" + (str)(current_hop)
        node_tracker[key] = topitem
        if current_hop >= k:
            continue
        current_node_neigh = G.neighbors(current_node)
        current_node_neigh.sort(key=lambda x: str(x))
        for nbr in current_node_neigh:
            node_tracker_current_node = node_tracker[key]
            if node_tracker_current_node.total_child_nodes >= avg_degree:
                break
            child_index = (
                node_tracker_current_node.index
                + (
                    level_min_max_index[current_hop][1]
                    - node_tracker_current_node.index
                )
                + (
                    node_tracker_current_node.index
                    - level_min_max_index[current_hop][0]
                )
                * avg_degree
                + node_tracker_current_node.total_child_nodes
                + 1
            )
            node_tracker_current_node.total_child_nodes = (
                node_tracker_current_node.total_child_nodes + 1
            )
            node_tracker[key] = node_tracker_current_node

            tree.add_vertex(ctr)
            tree.add_edge(current_node_new_name, ctr)
            tree.vs[ctr]["node"] = f"n{ctr}"
            tree.vs[ctr]["og_name"] = G.vs[nbr]["node"]
            tree.vs[ctr]["index"] = child_index  ##
            queue.append(
                qitem(
                    nbr,
                    ctr,
                    current_hop + 1,
                    child_index,
                    0,
                    node_tracker_current_node.index,
                )
            )
            ctr += 1

    # print("---------------")
    # for key, value in node_tracker.items():
    #     print(f"{key}: {value}")

    # for node in tree.vs:
    #     print(node["index"])

    return tree, node_tracker


def canonical_label(tree: ig.Graph, root: Optional[int] = None) -> str:
    r"""Computes the canonical label of tree"""
    if root is None:
        (root,) = np.where([v.indegree() == 0 for v in tree.vs])
        root = root.item()
    s = f"{tree.vs[root]['og_name']}"
    if len(tree.neighbors(root, mode="out")) != 0:
        child_labels = " ".join(
            sorted(
                canonical_label(tree, node) for node in tree.neighbors(root, mode="out")
            )
        )
        s = f"{s} {child_labels}"
    s = f"{s} $"
    return s


def featurize_label(label: str, features: List[List[float]]) -> str:
    r"""Computes featurized labels from node-name labels"""
    tokens = label.split()

    def feature(token):
        if token[0] == "n":
            return features[int(token[1:])]
        return token

    replaced = " ".join(
        "[" + " ".join(f"{k}" for k in feature(token)) + "]" for token in tokens
    )
    replaced = replaced.replace(" $", "$")
    replaced = replaced.replace("[$]", "$")
    replaced = replaced.replace(" $", "$")
    return replaced


def test():
    r"""Tests that the trees generated are same as when generated with
    networkx.
    Currently for Planetoid dataset.
    """
    data = Planetoid(root="data", name="cora")._data
    row, col = data.edge_index
    row, col = row.numpy(), col.numpy()
    n_vertices = data.x.shape[0]
    val = np.ones_like(row)
    adj = sp.coo_matrix((val, (row, col)), shape=(n_vertices, n_vertices))
    G = ig.Graph.Adjacency(adj, mode="undirected")
    G.vs["name"] = [f"n{i}" for i in range(len(G.vs))]
    node = 0
    hop = 2
    tree = khoptree(G, node, hop)

    # The following is just for verification
    def get_MPTree(Graph, root_node, hops):
        import networkx as nx
        from torch_geometric.utils import to_networkx

        Graph = to_networkx(Graph)
        assert hops < 5, (
            f"Number of hops {hops} is too computationally extensive for a proof of concept"
        )
        G = Graph.to_undirected()
        MPTree = nx.DiGraph()

        def inf_counter():
            ctr = 0
            while True:
                yield ctr
                ctr += 1

        ctr = inf_counter()
        start = root_node
        hop = 0
        Q = [(start, next(ctr), hop)]
        MPTree.add_node(Q[0][1], nodeorigid=start)
        while hop < hops and len(Q) > 0:
            top, topid, hop = Q.pop(0)
            if hop >= hops:
                continue
            neighbors = G.neighbors(top)
            for neighbor, new_id in zip(neighbors, ctr):
                MPTree.add_node(new_id, nodeorigid=neighbor)
                MPTree.add_edge(topid, new_id)
                Q.append((neighbor, new_id, hop + 1))
        return MPTree

    import networkx as nx
    import matplotlib.pyplot as plt

    tree2 = get_MPTree(data, node, hop)
    fig, ax = plt.subplots(2, 1)
    layout = tree.layout("tree")
    labels = [k.split("n")[-1] for k in tree.vs["og_name"]]
    ig.plot(
        tree,
        target=ax[0],
        layout=layout,
        vertex_label_size=10,
        vertex_label=labels,
        vertex_size=30,
        vertex_color="lightblue",
        edge_color="gray",
        bbox=(300, 300),
        margin=20,
    )
    pos = nx.nx_agraph.graphviz_layout(tree2, prog="dot")
    labels = {node: data["nodeorigid"] for node, data in tree2.nodes(data=True)}
    nx.draw(
        tree2,
        pos,
        ax=ax[1],
        with_labels=True,
        labels=labels,
        node_size=700,
        node_color="lightblue",
        edge_color="gray",
        font_size=10,
    )
    fig.savefig("test.png")  # look at this graph and verify visually


def test2():
    r"""Tests whether the canonical labelling is permutation invariant.
    Currently for Planetoid dataset.
    """
    data = Planetoid(root="data", name="cora")._data
    row, col = data.edge_index
    row, col = row.numpy(), col.numpy()
    n_vertices = data.x.shape[0]
    val = np.ones_like(row)
    adj = sp.coo_matrix((val, (row, col)), shape=(n_vertices, n_vertices))
    G = ig.Graph.Adjacency(adj, mode="undirected")
    G.vs["node"] = [f"n{i}" for i in range(len(G.vs))]
    hop = 2
    nnodes = data.x.shape[0]
    all_trees = (khoptree(G, node, hop) for node in range(nnodes))
    tree = next(all_trees)

    def permute_graph_with_attributes(graph):
        # Generate a random permutation of the vertex indices
        n = graph.vcount()
        perm = np.random.permutation(n)
        # Create a new graph with the same edges
        permuted_graph = ig.Graph(directed=True)
        permuted_graph.add_vertices(n)
        permuted_graph.add_edges(
            [(perm[edge.source], perm[edge.target]) for edge in graph.es]
        )
        # Permute node attributes
        inverse_perm = np.empty_like(perm)
        inverse_perm[perm] = np.arange(perm.size)
        for attr in graph.vs.attributes():
            permuted_graph.vs[attr] = [graph.vs[attr][i] for i in inverse_perm]
        return permuted_graph

    tree2 = permute_graph_with_attributes(tree)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1)
    layout = tree.layout("tree")
    labels = [k.split("n")[-1] for k in tree.vs["og_name"]]
    ig.plot(
        tree,
        target=ax[0],
        layout=layout,
        vertex_label_size=10,
        vertex_label=labels,
        vertex_size=30,
        vertex_color="lightblue",
        edge_color="gray",
        bbox=(300, 300),
        margin=20,
    )
    layout = tree2.layout("tree")
    labels = [k.split("n")[-1] for k in tree2.vs["og_name"]]
    ig.plot(
        tree2,
        target=ax[1],
        layout=layout,
        vertex_label_size=10,
        vertex_label=labels,
        vertex_size=30,
        vertex_color="lightblue",
        edge_color="gray",
        bbox=(300, 300),
        margin=20,
    )
    fig.savefig(
        "test2.png"
    )  # look at this graph to verify the two trees are isomorphic
    print(canonical_label(tree))
    print(canonical_label(tree2))  # these two should be the same


def sample_random_vectors(
    feature_size: int, avg_degree: int, k_hop: int, number_of_projectors: int
):
    """
    Generate l 2D matrices randomly sampled from a normal distribution with mean 0 and variance 1.
    """
    rows = sum(avg_degree**i for i in range(k_hop + 1))

    # Generate l matrices sampled from N(0, 1)
    matrices = np.random.normal(
        loc=3, scale=1, size=(rows, feature_size, number_of_projectors)
    )

    return matrices


def sort_nodes_and_index(items: List[qitem]) -> List[qitem]:
    items_with_hop = [item for item in items if item.hop > 0]

    grouped_by_parent: Dict[int, List[qitem]] = {}
    for item in items_with_hop:
        if item.parent_index not in grouped_by_parent:
            grouped_by_parent[item.parent_index] = []
        grouped_by_parent[item.parent_index].append(item)

    for parent, group in grouped_by_parent.items():
        original_indices = [item.index for item in group]
        original_indices.sort()
        # print("original_indices ", original_indices)
        group.sort(key=lambda x: str(x.v))

        for i, item in enumerate(group):
            # print(item.v, item.index, original_indices[i])
            item.index = original_indices[i]
    return items


def partition(list_bin_width, Bin_values, no_of_hash):
    summary_dict = {}
    print(list_bin_width)
    Bin_values = torch.tensor(Bin_values)
    for bin_width in list_bin_width:
        bias = torch.tensor(
            [random.uniform(-bin_width, bin_width) for i in range(no_of_hash)]
        )  # .to(device)
        temp = torch.floor((1 / bin_width) * (Bin_values + bias))  # .to(device)

        cluster, _ = torch.mode(temp, dim=1)
        dict_hash_indices = {}
        no_nodes = Bin_values.shape[0]
        for i in range(no_nodes):
            dict_hash_indices[i] = int(cluster[i])  # .to('cpu')
        summary_dict[bin_width] = dict_hash_indices

    return summary_dict


def get_key(val, g_coarsened):
    KEYS = []
    for key, value in g_coarsened.items():
        if val == value:
            KEYS.append(key)
    return len(KEYS), KEYS


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def validate(model, data):
    data = data  # .to(device)
    model.eval()
    # if args.model_type not in ['gcn','3wl']:
    #   pred = model(data.x, data.edge_index).argmax(dim=1)
    # elif args.model_type == '3wl':
    #   pred = model(data.x).argmax(1)
    # else:
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=1)

    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return acc


def val(model, data, model_type):
    data = data  # .to(device)
    model.eval()
    if model_type not in ["gcn", "3wl"]:
        pred = model(data.x, data.edge_index).argmax(dim=1)
    elif model_type == "3wl":
        pred = model(data.x).argmax(1)
    else:
        pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=1)

    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return acc


def train_on_original_dataset(
    data,
    num_classes,
    feature_size,
    hidden_units,
    learning_rate,
    decay,
    epochs,
    model_type="gcn",
):
    if model_type == "gin":
        model = GIN.GIN(feature_size, hidden_units, num_classes)
    elif model_type == "sage":
        model = GraphSage.GraphSAGE(feature_size, hidden_units, num_classes)
    elif model_type == "gat":
        model = GAT.GAT(feature_size, hidden_units, num_classes)
    elif model_type == "ugc":
        model = APPNP.Net(feature_size, hidden_units, num_classes)
    elif model_type == "3wl":
        model = WL_base_model.WL_BaseModel(feature_size, hidden_units, num_classes)
    else:
        model = GCN.GCN_(feature_size, hidden_units, num_classes)

    device = "cpu"
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=decay
    )
    #   test_split_percent = 0.2
    #   data = split(data,num_classes,test_split_percent)

    if data.edge_attr == None:
        edge_weight = torch.ones(data.edge_index.size(1))
        data.edge_attr = edge_weight

    for epoch in range(epochs):
        optimizer.zero_grad()
        if model_type not in ["gcn", "3wl"]:
            out = model(data.x, data.edge_index)
        elif model_type == "3wl":
            out = model(data.x)
        else:
            out = model(data.x, data.edge_index, data.edge_attr.float())

        pred = out.argmax(1)
        criterion = torch.nn.NLLLoss()

        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        best_val_acc = 0

        val_acc = val(model, data, model_type)
        if best_val_acc < val_acc:
            torch.save(model, "full_best_model.pt")
            best_val_acc = val_acc

        if epoch % 100 == 0:
            print(
                "In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})".format(
                    epoch, loss, val_acc, best_val_acc
                )
            )

    model = torch.load("full_best_model.pt")
    model.eval()

    if model_type not in ["gcn", "3wl"]:
        pred = model(data.x, data.edge_index).argmax(dim=1)
    elif model_type == "3wl":
        pred = model(data.x).argmax(dim=1)
    else:
        pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=1)

    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())

    incorrect_indices = (pred[data.test_mask] != data.y[data.test_mask]).nonzero()

    # Convert the indices to a list
    incorrect_indices_list = incorrect_indices.view(-1).tolist()

    print("--------------------------")
    print("Accuracy on test data {:.3f}".format(acc * 100))

    return incorrect_indices_list


from torch_geometric.utils import (
    to_networkx,
    to_undirected,
    remove_self_loops,
    coalesce,
)
import networkx as nx


def build_exact_k_hop_edge_index(data, k: int):
    """
    Returns edge_index' where (u,v) is an edge iff shortest-path distance in the ORIGINAL graph is exactly k.
    Assumes an undirected graph setting (typical for many PyG node-classification datasets).
    """
    assert k >= 1, "k must be >= 1"
    num_nodes = data.num_nodes

    # Convert to NetworkX for shortest-path computation
    G = to_networkx(data)

    # Build k-hop graph
    Hk = nx.Graph()
    Hk.add_nodes_from(range(num_nodes))

    dist_dict = dict(nx.all_pairs_shortest_path_length(G, cutoff=k))
    for u, dmap in dist_dict.items():
        for v, d in dmap.items():
            if d == k and u < v:
                Hk.add_edge(u, v)

    # Convert back to edge_index
    edges_k = list(Hk.edges())

    edge_index_k = torch.tensor(edges_k, dtype=torch.long).t().contiguous()
    edge_index_k = to_undirected(edge_index_k, num_nodes=num_nodes)

    edge_index_k, _ = remove_self_loops(edge_index_k)
    edge_index_k, _ = coalesce(edge_index_k, None, num_nodes, num_nodes)

    return edge_index_k


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
    print(len(set(y.item() for y in data.y)))

    ########################################################## preprocessing data for heterophilic graphs
    if args.heterophilic_hop_k != 0:
        heterophilic_hop_k = args.heterophilic_hop_k
        data.edge_index = build_exact_k_hop_edge_index(data, heterophilic_hop_k)

    ##########################################################

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

    # time1 = time.time()
    # incorrect_index = train_on_original_dataset(data,dataset.num_classes,dataset.num_features,args.hidden_units,args.lr,args.decay,args.epochs, args.model_type)
    # time2 = time.time()
    # print("time taken to train GCN ", time2 - time1)
    # # print(incorrect_index)
    # # print(len(incorrect_index))
    # exit(1)

    n_vertices = data.x.shape[0]
    train_size = int(0.6 * n_vertices)
    # train_nodes = range(train_size)
    row, col = data.edge_index
    row, col = row.numpy(), col.numpy()
    val = np.ones_like(row)
    adj = sp.coo_matrix((val, (row, col)), shape=(n_vertices, n_vertices))
    G = ig.Graph.Adjacency(adj, mode="undirected")
    G.vs["node"] = [f"n{i}" for i in range(len(G.vs))]
    hop = 2

    avg_degree = 5
    level_min_max_index = {}
    for level in range(hop):
        if level == 0:
            level_min_max_index[level] = [0, 0]
        else:
            level_min_max_index[level] = [
                level_min_max_index[level - 1][1] + 1,
                (level_min_max_index[level - 1][1] + 1) * avg_degree,
            ]

    # train_trees, node_trackers = zip(*(khoptree(G, node, hop, level_min_max_index, avg_degree) for node in train_nodes))

    ## we have hashed all the nodes including test(training is being done by supernodes which only contains train nodes)
    train_trees, node_trackers = zip(
        *(
            khoptree(G, node, hop, level_min_max_index, avg_degree)
            for node in range(data.x.shape[0])
        )
    )
    # train_trees = list(train_trees)
    node_trackers = list(node_trackers)
    train_labels = (canonical_label(tree) for tree in train_trees)
    print(list(train_labels)[0])
    features = data.x.numpy()
    # featurized_labels = (featurize_label(label, features) for label in train_labels)
    # print(next(featurized_labels))

    # PCA reduce original features to 50 dims before augmentation
    pca_components = 50
    pca = PCA(n_components=pca_components, random_state=42)
    features_pca = pca.fit_transform(features)

    # Build augmented features for all nodes using PCA-reduced features
    filler_delim = None  # use default (-2) inside the function
    hop_delim = None  # use default (-1) inside the function
    augmented_feature_list = []
    for nt in node_trackers:
        vec = build_augmented_feature_for_tree(
            nt,
            features_pca,
            avg_degree=avg_degree,
            k_hop=hop,
            filler_value=0.0,
            hop_delimiter=hop_delim,
            filler_delimiter=filler_delim,
        )
        augmented_feature_list.append(vec)
    augmented_features = np.vstack(augmented_feature_list)
    print("Augmented feature matrix shape:", augmented_features.shape)

    # Run GPU-accelerated classical ML experiments and save CSV
    results_df = run_classical_ml(
        augmented_features,
        data,
        # output_csv_path=f"{args.dataset_name}_classical_ml_results.csv",
        # output_csv_path=f"{args.dataset_name}_classical_ml_results_augmented_features_1.csv",
        output_csv_path=f"{args.dataset_name}_classical_ml_results_with_hetero_hops_{heterophilic_hop_k}.csv",
    )
    print(results_df)

    for node_tracker in node_trackers[0].values():
        print(node_tracker)

    print(len(node_trackers))

    exit()

    ##########################################################

    number_of_projectors = 10
    random_matrices = sample_random_vectors(
        features.shape[1], avg_degree, hop, number_of_projectors
    )
    print("random_matrices ", random_matrices.shape)

    summary = {}
    counter = 0

    for node_tracker in node_trackers:
        sorted_nodes_and_index = (
            node_tracker.values()
        )  # sort_nodes_and_index(node_tracker.values())
        # print("sorted_nodes_and_index ",sorted_nodes_and_index)
        feature_node_index = []
        random_vector_index = []
        accumulated_dot_products = np.zeros(number_of_projectors)

        for item in sorted_nodes_and_index:
            current_node_feature = features[item.v]
            current_node_random_projecters = random_matrices[item.index, :, :].T
            # print("current_node_random_projecters ",current_node_random_projecters.shape, current_node_feature.shape)

            dot_products = np.dot(current_node_random_projecters, current_node_feature)
            # print("accumulated_dot_products ", accumulated_dot_products.shape, dot_products.shape)
            accumulated_dot_products += dot_products

        # print("accumulated_dot_products ", accumulated_dot_products)
        summary[counter] = accumulated_dot_products
        counter += 1

    print(np.array(list(summary.values())))
    print(np.array(list(summary.values())).shape)
    print(counter)

    list_bin_width = [0.2]
    current_bin_width_summary = partition(
        list_bin_width, np.array(list(summary.values())), number_of_projectors
    )[list_bin_width[0]]

    values = current_bin_width_summary.values()
    # print(values)
    unique_values = set(values)
    rr = 1 - len(unique_values) / len(values)
    print(
        f"Graph reduced by: {rr * 100} percent.\nWe now have {len(unique_values)} supernode, starting nodes were: {len(values)}"
    )
    dict_blabla = {}
    C_diag = torch.zeros(len(unique_values))  # , device= device)
    help_count = 0

    for v in unique_values:
        C_diag[help_count], dict_blabla[help_count] = get_key(
            v, current_bin_width_summary
        )
        help_count += 1

    representatives = [v[0] for _, v in dict_blabla.items()]
    # print("representatives ", representatives)

    from torch_geometric.utils import subgraph, to_dense_adj

    data_coarsen_ei, _ = subgraph(representatives, data.edge_index, relabel_nodes=True)
    data_coarsen_ft = data.x[representatives]
    data_coarsen_ys = data.y[representatives]
    data_coarsen = Data(
        x=data_coarsen_ft, edge_index=data_coarsen_ei, y=data_coarsen_ys
    )

    adj_dense = to_dense_adj(data_coarsen.edge_index, edge_attr=data.edge_attr)[0]
    edge_weight = adj_dense[np.nonzero(adj_dense)]
    # edge_features = edge_weight#torch.from_numpy(edge_weight)

    data_coarsen.edge_attr = edge_weight

    ######################

    # P_hat = torch.zeros((data.num_nodes, len(unique_values)))#, device= device)
    # print("P_hat ", P_hat.shape)
    # zero_list = torch.ones(len(unique_values), dtype=torch.bool)
    #
    # for x in dict_blabla:
    #     if len(dict_blabla[x]) == 0:
    #         print("zero element in this supernode",x)
    #     for y in dict_blabla[x]:
    #         P_hat[y,x] = 1
    #         zero_list[x] = zero_list[x] and (not (data.train_mask)[y])
    #
    # P_hat = P_hat.to_sparse()
    #
    # ## double check normalization
    # #dividing by number of elements in each supernode to get average value
    # P = torch.sparse.mm(P_hat,(torch.diag(torch.pow(C_diag, -1/2))))
    #
    # ## P = torch.sparse.mm(torch.diag(torch.pow(C_diag, -1/2)),torch.sparse.mm(P_hat,torch.diag(torch.pow(C_diag, -1/2))))
    # ####
    #
    # print(P.shape)
    #
    # cor_feat = (torch.sparse.mm((torch.t(P)), torch.tensor(features)))#.to_sparse()
    # i = data.edge_index
    # v = torch.ones(data.edge_index.shape[1])
    # shape = torch.Size([data.x.shape[0],data.x.shape[0]])
    # g_adj_tens = torch.sparse.FloatTensor(i, v, torch.Size(shape))#.to(device = device)
    # g_coarse_adj = torch.sparse.mm(torch.t(P_hat) , torch.sparse.mm( g_adj_tens , P_hat))
    #
    # ## double check
    # C_diag_matrix = np.diag(np.array(C_diag.to('cpu'), dtype = np.float32))
    # #print("number of edges in the coarsened graph ",np.count_nonzero(g_coarse_adj.to_dense().to('cpu').numpy())/2)
    #
    # g_coarse_dense = g_coarse_adj.to_dense().to('cpu').numpy() + C_diag_matrix - np.identity(C_diag_matrix.shape[0], dtype = np.float32)
    # ####
    #
    # edge_weight = g_coarse_dense[np.nonzero(g_coarse_dense)]
    # edges_src = torch.from_numpy((np.nonzero(g_coarse_dense))[0])
    # edges_dst = torch.from_numpy((np.nonzero(g_coarse_dense))[1])
    # edge_index_corsen = torch.stack((edges_src, edges_dst))
    # edge_features = torch.from_numpy(edge_weight)
    #
    num_classes = dataset.num_classes
    # Y = np.array(data.y.cpu())
    # Y = one_hot(Y,num_classes)#.to(device)
    # Y[~data.train_mask] = torch.Tensor([0 for _ in range(num_classes)])#.to(device)
    # labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P).double() , Y.double()).double() , 1)#.to(device)
    #
    # # deleting unused variables
    # del C_diag_matrix
    # del g_coarse_adj
    # del edge_weight
    # del edges_dst
    # del i
    # del v
    #
    # data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
    # data_coarsen.edge_attr = edge_features

    num_run = 1
    time5 = time.time()
    time_taken_to_train_gcn = []
    all_acc = []
    for i in range(num_run):
        global_best_val = 0
        global_best_test = 0
        best_val_acc = 0
        best_epoch = 0

        hidden_units = args.hidden_units
        learning_rate = args.lr
        decay = args.decay
        epochs = args.epochs

        feature_size = features.shape[1]
        if args.model_type == "gin":
            model = GIN.GIN(feature_size, hidden_units, num_classes)
        elif args.model_type == "sage":
            model = GraphSage.GraphSAGE(feature_size, hidden_units, num_classes)
        elif args.model_type == "gat":
            model = GAT.GAT(feature_size, hidden_units, num_classes)
        elif args.model_type == "ugc":
            model = APPNP.Net(feature_size, hidden_units, num_classes)
        elif args.model_type == "3wl":
            model = WL_base_model.WL_BaseModel(feature_size, hidden_units, num_classes)
        else:
            model = GCN.GCN_(feature_size, hidden_units, num_classes)

        device = "cpu"
        # device = 'cpu'
        model = model.to(device)
        data = data.to(device)
        data_coarsen = data_coarsen.to(device)
        edge_weight = torch.ones(data_coarsen.edge_index.size(1))
        decay = decay
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=decay
        )

        for epoch in range(epochs):
            optimizer.zero_grad()

            if args.model_type not in ["gcn", "3wl"]:
                out = model(data_coarsen.x, data_coarsen.edge_index)
            elif args.model_type == "3wl":
                out = model(data_coarsen.x)
            else:
                out = model(
                    data_coarsen.x,
                    data_coarsen.edge_index,
                    data_coarsen.edge_attr.float(),
                )

            # out = model(data_coarsen.x, data_coarsen.edge_index,data_coarsen.edge_attr.float())
            # out = model(data_coarsen.x, data_coarsen.edge_index)

            pred = out.argmax(1)
            criterion = torch.nn.NLLLoss()
            # print(out.shape)
            loss = criterion(out[~zero_list], data_coarsen.y[~zero_list])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(data.x.dtype)
            # print(data.edge_index.dtype)
            # print(data.edge_attr.dtype)
            val_acc = validate(model, data)

            if best_val_acc < val_acc:
                torch.save(model, "best_model.pt")
                best_val_acc = val_acc
                best_epoch = epoch

            if epoch % 100 == 0:
                print(
                    "In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})".format(
                        epoch, loss, val_acc, best_val_acc
                    )
                )

        time6 = time.time()
        # print('diff b/w t6 and t5 {}'.format(time6-time5))
        time_taken_to_train_gcn.append(time6 - time5)
        model = torch.load("best_model.pt")
        model.eval()
        data = data.to(device)

        if args.model_type not in ["gcn", "3wl"]:
            pred = model(data.x, data.edge_index).argmax(dim=1)
        elif args.model_type == "3wl":
            pred = model(data.x).argmax(dim=1)
        else:
            pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=1)

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()

        acc = int(correct) / int(data.test_mask.sum())

        # print('diff b/w t7 and t5 {}'.format(time7-time5))
        all_acc.append(acc)

    print("ratio ", rr)
    print(
        "ave_acc: {:.4f}".format(np.mean(all_acc)), "+/- {:.4f}".format(np.std(all_acc))
    )
    print(
        "ave_time: {:.4f}".format(np.mean(time_taken_to_train_gcn)),
        "+/- {:.4f}".format(np.std(time_taken_to_train_gcn)),
    )
    # print("he_error_list ",he_error_list)
    # print("ree_error_list ",ree_error_list)
    # print("dirichlet_energy_list ",dirichlet_energy_list)


def build_augmented_feature_for_tree(
    node_tracker: Dict[str, qitem],
    features: np.ndarray,
    avg_degree: int,
    k_hop: int,
    filler_value: float = 0.0,
    hop_delimiter: Optional[np.ndarray] = None,
    filler_delimiter: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Create hop-delimited augmented feature vector for a single computational tree.

    Rules implemented from the requirements:
    - Define a common delimiter to separate hops (hop_delimiter), appended between hops.
    - For each hop, concatenate node features in the order of `new_name`.
    - Ensure each hop has a fixed capacity of avg_degree**hop (with hop 0 fixed to 1).
      Pad missing slots with the filler delimiter vector (or filler_value if no delimiter given).
    - Concatenate all hops (with hop delimiters in between) into one 1D vector.
    """
    feature_dim = features.shape[1]

    # Default delimiters
    if hop_delimiter is None:
        hop_delimiter = np.full((feature_dim,), -1.0, dtype=features.dtype)
    if filler_delimiter is None:
        filler_delimiter = np.full((feature_dim,), -2.0, dtype=features.dtype)

    # Group items by hop
    hop_to_items: Dict[int, List[qitem]] = {}
    for item in node_tracker.values():
        if item.hop not in hop_to_items:
            hop_to_items[item.hop] = []
        hop_to_items[item.hop].append(item)

    augmented_chunks: List[np.ndarray] = []
    for hop in range(0, k_hop + 1):
        items_in_hop = hop_to_items.get(hop, [])
        items_in_hop.sort(key=lambda it: it.new_name)

        expected_count = int(avg_degree**hop)
        if hop == 0:
            expected_count = 1

        # Take up to expected_count real nodes
        real_features: List[np.ndarray] = [
            features[it.v] for it in items_in_hop[:expected_count]
        ]
        actual_count = len(real_features)

        # Pad to expected_count using filler delimiter vector
        if actual_count < expected_count:
            num_to_fill = expected_count - actual_count
            pad_block = np.tile(filler_delimiter.reshape(1, -1), (num_to_fill, 1))
            hop_matrix = (
                np.vstack(real_features + [pad_block])
                if actual_count > 0
                else pad_block
            )
        else:
            hop_matrix = np.vstack(real_features)

        hop_flat = hop_matrix.reshape(-1)
        augmented_chunks.append(hop_flat)

        # Add hop delimiter between hops (not after the last one)
        if hop < k_hop:
            augmented_chunks.append(hop_delimiter.reshape(-1))

    return np.concatenate(augmented_chunks, axis=0)


if __name__ == "__main__":
    # test()
    # test2()
    main()

    # items = [
    # qitem(v=1623, new_name=0, hop=0, index=0, total_child_nodes=5, parent_index=0),
    # qitem(v=102, new_name=1, hop=1, index=1, total_child_nodes=5, parent_index=1),
    # qitem(v=306, new_name=16, hop=2, index=2, total_child_nodes=0, parent_index=1),
    # qitem(v=112, new_name=2, hop=1, index=17, total_child_nodes=4, parent_index=1),
    # qitem(v=109, new_name=20, hop=2, index=21, total_child_nodes=0, parent_index=4),
    # # Add other items as necessary...
    # ]
