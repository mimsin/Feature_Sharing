import pandas as pd
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv #GATConv
import seaborn as sns

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index,i):
        if (i==0):
            # First Message Passing Layer (Transformation)
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            return x
        else:
            # Second Message Passing Layer
            x = self.conv2(x, edge_index)
            #x = x.relu()
            #x = F.dropout(x, p=0.5, training=self.training)
            # Output layer
            x = F.softmax(self.out(x), dim=1)
            return x


