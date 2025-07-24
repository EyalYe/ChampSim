# model.py
import torch.nn as nn
import torch.nn.functional as F
import torch

class PrefetchCNN(nn.Module):
    def __init__(self, n_pcs, n_deltas,
                 pc_emb=64, delta_emb=16,
                 hidden=128, kernel=3, hist=16,
                 n_classes=128, dropout=0.2):
        super().__init__()
        self.pc_embed    = nn.Embedding(n_pcs, pc_emb)
        self.delta_embed = nn.Embedding(n_deltas, delta_emb)
        in_ch = pc_emb + delta_emb

        self.conv1 = nn.Conv1d(in_ch, hidden, kernel, padding=kernel//2)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel//2)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel, padding=kernel//2)
        self.bn3   = nn.BatchNorm1d(hidden)

        self.drop = nn.Dropout(dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc   = nn.Linear(hidden, n_classes)

    def forward(self, pc_seq, delta_seq):
        p = self.pc_embed(pc_seq)
        d = self.delta_embed(delta_seq)
        x = torch.cat([p, d], dim=2)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

