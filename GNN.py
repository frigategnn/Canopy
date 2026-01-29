"""
Single file to run GCN, SAGE, GIN, and GAT models on all datasets.
Each model has 2 layers. GPU enabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from pathlib import Path
import numpy as np
from termcolor import colored
import json
from datetime import datetime
import os


# ============================================================================
# Model Definitions (2 layers each)
# ============================================================================

class SimpleGCN(nn.Module):
    """Simple 2-layer GCN model"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SimpleSAGE(nn.Module):
    """Simple 2-layer GraphSAGE model"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(SimpleSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SimpleGIN(nn.Module):
    """Simple 2-layer GIN model"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(SimpleGIN, self).__init__()
        # GIN uses MLP networks
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1, train_eps=True)
        self.conv2 = GINConv(nn2, train_eps=True)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


class SimpleGAT(nn.Module):
    """Simple 2-layer GAT model"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.5):
        super(SimpleGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train(model, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
    return acc


def run_model_on_dataset(model_class, model_name, dataset_dict, device, config):
    """Train and evaluate a model on a dataset"""
    
    # Extract data from dataset_dict
    graph = dataset_dict["graph"]
    labels = dataset_dict["label"].squeeze()
    
    # Create PyG Data object
    edge_index = graph["edge_index"]
    node_feat = graph["node_feat"]
    num_nodes = graph["num_nodes"]
    
    # Make graph undirected and add self-loops
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    # Create splits (60/20/20)
    gen = torch.Generator().manual_seed(config["seed"])
    perm = torch.randperm(num_nodes, generator=gen)
    train_end = int(0.6 * num_nodes)
    val_end = int(0.8 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True
    
    data = Data(x=node_feat, edge_index=edge_index, y=labels)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Move to device
    data = data.to(device)
    
    # Get dimensions
    in_channels = node_feat.shape[1]
    out_channels = int(labels.max().item() + 1)
    hidden_channels = config["hidden_channels"]
    
    # Initialize model
    if model_name.lower() == "gat":
        model = model_class(in_channels, hidden_channels, out_channels, 
                          heads=config.get("gat_heads", 1), 
                          dropout=config["dropout"]).to(device)
    else:
        model = model_class(in_channels, hidden_channels, out_channels, 
                          dropout=config["dropout"]).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config["lr"], 
                               weight_decay=config["weight_decay"])
    criterion = nn.NLLLoss()
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    
    for epoch in range(config["epochs"]):
        loss = train(model, data, data.train_mask, optimizer, criterion)
        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
        
        if epoch % config["display_step"] == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | "
                  f"Best Val: {best_val_acc:.4f} | Best Test: {best_test_acc:.4f}")
    
    return {
        "model": model_name,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "final_val_acc": val_acc,
        "final_test_acc": test_acc
    }


# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Configuration
    config = {
        "seed": 42,
        "hidden_channels": 256,
        "lr": 0.001,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "epochs": 1000,
        "display_step": 50,
        "gat_heads": 1,
    }
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(f"Using device: {colored(device, 'green', attrs=['bold'])}")
    # if torch.cuda.is_available():
    #     print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    
    # # Find dataset directory
    # possible_paths = [
    #     Path("../save_ds/save_dataset_with_splits"),
    #     Path("../../save_ds/save_dataset_with_splits"),
    #     Path("./save_dataset_with_splits"),
    #     Path("../save_dataset_with_splits"),
    # ]
    
    # dataset_root = None
    # for path in possible_paths:
    #     if path.exists():
    #         dataset_root = path
    #         break
    
    # if dataset_root is None:
    #     print(f"Error: Dataset directory not found. Tried: {possible_paths}")
    #     return

    dataset_root = Path("./save_ds/save_dataset_with_splits")
    print(f"Dataset directory: {colored(str(dataset_root), 'blue', attrs=['bold'])}\n")
    
    # Model classes
    models = {
        "GCN": SimpleGCN,
        "SAGE": SimpleSAGE,
        "GIN": SimpleGIN,
        "GAT": SimpleGAT,
    }
    
    # Results storage
    all_results = []
    
    # Iterate through all datasets
    dataset_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.stem
        dataset_file = dataset_dir / f"{dataset_name}.pth"
        
        if not dataset_file.exists():
            print(f"Warning: {dataset_file} not found, skipping {dataset_name}")
            continue
        
        print("=" * 80)
        print(f"Dataset: {colored(dataset_name, 'red', attrs=['bold'])}")
        print("=" * 80)
        
        # Load dataset
        try:
            dataset_dict = torch.load(dataset_file, map_location=device)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue
        
        # Run each model
        for model_name, model_class in models.items():
            print(f"\n{'-' * 80}")
            print(f"Running {colored(model_name, 'blue', attrs=['bold'])} on {dataset_name}")
            print(f"{'-' * 80}")
            
            try:
                result = run_model_on_dataset(
                    model_class, model_name, dataset_dict, device, config
                )
                result["dataset"] = dataset_name
                all_results.append(result)
                
                print(f"\n✓ {model_name} on {dataset_name}:")
                print(f"  Best Val Acc: {result['best_val_acc']:.4f}")
                print(f"  Best Test Acc: {result['best_test_acc']:.4f}")
                
            except Exception as e:
                print(f"\n✗ Error running {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "=" * 80 + "\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_all_models_{timestamp}.json"
    
    # Convert to JSON-serializable format
    results_json = []
    for r in all_results:
        results_json.append({
            "dataset": r["dataset"],
            "model": r["model"],
            "best_val_acc": float(r["best_val_acc"]),
            "best_test_acc": float(r["best_test_acc"]),
            "best_epoch": int(r["best_epoch"]),
            "final_val_acc": float(r["final_val_acc"]),
            "final_test_acc": float(r["final_test_acc"]),
        })
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("=" * 80)
    print(f"Results saved to: {colored(results_file, 'green', attrs=['bold'])}")
    print("=" * 80)
    
    # Print summary
    print("\nSummary:")
    print("-" * 80)
    print(f"{'Dataset':<20} {'Model':<8} {'Val Acc':<10} {'Test Acc':<10}")
    print("-" * 80)
    for r in results_json:
        print(f"{r['dataset']:<20} {r['model']:<8} {r['best_val_acc']:<10.4f} {r['best_test_acc']:<10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

