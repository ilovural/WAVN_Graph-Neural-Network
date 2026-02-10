'''
This file defines the GNN used to learn from the image-pair graphs
Notes: the class GNN is a simple 2-layer GCN that is followed by a global pooling & linear classification head
Global Pooling - technique that aggregates the features from all of the nodes in a graph into a single, fixed-size vector that represents the entire graph
Linear Classification Head - a simple, fully connected layer that takes the final learned node or graph representations and maps them to the desired output classes
'''
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    #initialising model
    def __init__(self, inChannels, hiddenChannels, numClasses, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(inChannels, hiddenChannels)  # Layer 1
        self.conv2 = GCNConv(hiddenChannels, hiddenChannels)  # Layer 2
        self.dropout = dropout
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hiddenChannels * 2, hiddenChannels),
            torch.nn.ReLU(),  # Non-linearity
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hiddenChannels, numClasses)
        )

    # Forward method must be at class level, not inside __init__
    def forward(self, x, edge_index):
        #node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        #edge embeddings
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=1)

        #edge-level logits
        return self.edge_mlp(edge_features)
