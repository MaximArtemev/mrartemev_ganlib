"""
Predefined Networks
"""

import torch
from torch import nn
from .networks_utils import ResidualBlock, init_weights


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, in_features, out_features, hidden_features=32, depth=4, context=False):
        """ context  - int, False or zero if None""" 
        super().__init__()
        assert depth >= 2
        self.net = [nn.Linear(in_features + int(context), hidden_features),
                    nn.LeakyReLU(0.01)]
        for _ in range(depth-2):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.LeakyReLU(0.01))
        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)
        self.apply(init_weights)

    def forward(self, x, context=None):
        if context is not None:
            return self.net(torch.cat([x, context], dim=1))
        return self.net(x)


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(self, in_features, out_features, hidden_features=32, depth=4, context=False):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(in_features + int(context), hidden_features),
            nn.BatchNorm1d(hidden_features, eps=1e-3),
            nn.LeakyReLU(0.01)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(
                features=hidden_features,
                context=context,
            ) for _ in range(depth//2)
        ])
        self.final_layer = nn.Linear(hidden_features, out_features)
        self.apply(init_weights)

    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=1)
        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x, context=context)
        outputs = self.final_layer(x)
        return outputs
