# model/minimal_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def create_dynamic_edges(batch_data, k=4):
    batch_size, num_nodes, num_features = batch_data.shape
    edge_index = []
    k = min(k, num_nodes - 1)

    for batch in range(batch_size):
        sample = batch_data[batch]
        sim_matrix = torch.cdist(sample, sample, p=2)
        knn_edges = torch.topk(sim_matrix, k + 1, largest=False)[1]

        for i in range(num_nodes):
            for j in range(1, k + 1):
                src = batch * num_nodes + i
                tgt = batch * num_nodes + knn_edges[i, j].item()
                edge_index.append([src, tgt])
                edge_index.append([tgt, src])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

class GCN_TCN_CapsNet(nn.Module):
    def __init__(self, input_dim=3, num_classes=6, num_nodes=128):
        super(GCN_TCN_CapsNet, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes

        self.gcn1 = GCNConv(input_dim, 32)
        self.gcn2 = GCNConv(32, 64)
        self.dropout_gcn = nn.Dropout(0.3)

        self.conv1 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.dropout_tcn = nn.Dropout(0.3)

        self.caps1 = nn.Linear(256 * (num_nodes // 2), 128)
        self.caps2 = nn.Linear(128, num_classes)
        self.dropout_caps = nn.Dropout(0.3)

    def forward(self, x):
        batch_size, num_nodes, channels = x.shape
        device = x.device  # Get the device of the input tensor

        edge_index = create_dynamic_edges(x, k=4).to(device)  # Move edge_index to the same device

        x = x.reshape(batch_size * num_nodes, channels)
        x = F.relu(self.gcn1(x, edge_index))
        x = self.dropout_gcn(F.relu(self.gcn2(x, edge_index)))

        x = x.view(batch_size, num_nodes, -1).permute(0, 2, 1)
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(self.dropout_tcn(F.relu(self.batchnorm2(self.conv2(x)))))

        x = x.view(batch_size, -1)
        x = self.dropout_caps(F.relu(self.caps1(x)))
        return self.caps2(x)
