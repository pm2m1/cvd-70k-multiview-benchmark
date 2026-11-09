"""Configurable five-view encoders and a capacity-matched ordinary MLP."""

from __future__ import annotations

import torch
from torch import nn


def block(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, outputs), nn.ReLU(), nn.LayerNorm(outputs))


class FiveViewMLP(nn.Module):
    def __init__(self, view_dims=(2, 2, 3, 6, 3), embedding=16, fusion=(64, 32)):
        super().__init__()
        if len(view_dims) != 5 or min(*view_dims, embedding, *fusion) < 1:
            raise ValueError("Five nonempty views and positive widths required")
        self.dims = list(view_dims)
        self.encoders = nn.ModuleList([block(d, embedding) for d in view_dims])
        self.fusion1 = block(5 * embedding, fusion[0])
        self.fusion2 = block(fusion[0], fusion[1])
        self.head = nn.Linear(fusion[1], 1)

    def forward(self, x):
        views = torch.split(x, self.dims, dim=1)
        z = torch.cat([encoder(v) for encoder, v in zip(self.encoders, views, strict=True)], 1)
        return self.head(self.fusion2(self.fusion1(z))).squeeze(-1)


class FlatMLP(nn.Module):
    def __init__(self, inputs=16, hidden=(96, 64)):
        super().__init__()
        if min(inputs, *hidden) < 1:
            raise ValueError("Positive widths required")
        layers = []
        for d in hidden:
            layers.append(block(inputs, d))
            inputs = d
        self.net = nn.Sequential(*layers, nn.Linear(inputs, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
