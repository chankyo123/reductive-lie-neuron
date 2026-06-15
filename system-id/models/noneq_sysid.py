import torch
import torch.nn as nn


class NonEqNet(nn.Module):
    def __init__(self, n=3, hidden_dim=64):
        super().__init__()
        d = n * n
        self.encoder = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, d),
        )
        self.n = n

    def forward(self, x):
        # x: [B, N, 3, 3]
        B, N, n, _ = x.shape
        x_flat = x.reshape(B, N, -1)
        encoded = self.encoder(x_flat)
        pooled = encoded.mean(dim=1)
        out = self.decoder(pooled)
        return out.reshape(B, self.n, self.n)
