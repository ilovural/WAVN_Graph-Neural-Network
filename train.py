'''
This file trains the Graph Neural Network model on the Gazebo 'images' dataset. 
Notes: The model loads the dataset from labels.csv and then initialises the GNN model. 
After initialising, it trains with x-entropy loss to evaluate accuracy. 
'''
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import time
from dataset import NavGraphDataset
from GNNmodel import GNN  

CSVpath = "labels.csv"
imageDirectory = "images/rgb"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#experiment configuration
experiment_configuration = {
    "epochs": 1000,
    "learningRate": 1e-3,
    "hiddenDimension": 96,
    "trainSplit": 0.7,  #not used with k-fold
}

EPOCHS = experiment_configuration["epochs"]
LR = experiment_configuration["learningRate"]
hiddenDimension = experiment_configuration["hiddenDimension"]


def train_with_cross_validation(k_folds=5):
    dataset = NavGraphDataset(CSVpath, imageDirectory, device)
    data = dataset.get_graph().to(device)

    num_edges = data.edge_attr.size(0)
    edge_indices = torch.arange(num_edges)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(edge_indices)):
        print(f"\nFold {fold+1}/{k_folds}")

        train_idx = torch.tensor(train_idx, device=device)
        val_idx = torch.tensor(val_idx, device=device)

        model = GNN(
            inChannels=data.x.shape[1],
            hiddenChannels=hiddenDimension,
            numClasses=len(dataset.LabelMap)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss()
        best_val_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()

            #forward pass
            out = model(data.x, data.edge_index)

            loss = criterion(out[train_idx], data.edge_attr[train_idx])
            loss.backward()
            optimizer.step()

            #validation
            model.eval()
            with torch.no_grad():
                val_preds = out[val_idx].argmax(dim=1)
                val_acc = ((val_preds == data.edge_attr[val_idx]).float().mean().item()) * 100

            best_val_acc = max(best_val_acc, val_acc)

            if epoch % 50 == 0:
                print(f"Epoch {epoch:4d} | Loss {loss.item():.4f} | Val Acc {val_acc:.2f}%")

        print(f"Best Val Accuracy (Fold {fold+1}): {best_val_acc:.2f}%")
        fold_accuracies.append(best_val_acc)

    print("\nCross-Validation Results")
    print("Fold Accuracies:", fold_accuracies)
    print(
        f"Mean Accuracy: {sum(fold_accuracies)/len(fold_accuracies):.2f}% ± "
        f"{torch.std(torch.tensor(fold_accuracies)):.2f}"
    )


def evaluate(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)
    return 100.0 * correct / total if total > 0 else 0


if __name__ == "__main__":
    start_time = time.perf_counter()
    train_with_cross_validation(k_folds=5)
    end_time = time.perf_counter()
    print("Total runtime: ", end_time - start_time, " seconds")

"""
RESULTS: 10 February 2026

Cross-Validation Results
Fold Accuracies: 
[64.45473432540894, 
63.046836853027344, 
64.02470469474792, 
62.68656849861145, 
62.223368883132935]
Mean Accuracy: 63.29% ± 0.93
Total runtime:  678.554190334049  seconds
"""
