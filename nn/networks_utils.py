"""
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(self, features, context=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features + int(context), features),
            nn.BatchNorm1d(features, eps=1e-3),
            nn.LeakyReLU(0.05),
            nn.Dropout(p=0.3),
            nn.Linear(features, features),
            nn.BatchNorm1d(features, eps=1e-3),
            nn.LeakyReLU(0.05),
        )

    def forward(self, x, context=None):
        if context is not None:
            return self.net(torch.cat([x, context], dim=1)) + x
        return self.net(x) + x


def gradient_penalty(critic, real_data, fake_data, context):
    """
    Computes Gradient Penalty in random interpolates, in its classic form:
    (|∇(D(x)|^2 - 1)^2, x is interpolated between a real and a generated sample
    Args:
      critic: a torch model which gradient needs to be penalised
      real_data[batch_size, n_features]: a sample of real data
      fake_data[batch_size, n_features]: a sample of fake data
    Returns:
      torch.Tensor, scalar, gradient penalty evalute
    """
    assert real_data.shape == fake_data.shape
    alpha = torch.rand_like(real_data)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)

    disc_interpolates = critic(interpolates, context)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    return (gradients.norm(2, dim=1) - 1) ** 2


def dragan_gradient_penalty(critic, real_data, context):
    """Calculates the gradient penalty loss for DRAGAN"""
    # Random weight term for interpolation
    alpha = torch.rand_like(real_data)
    interpolates = alpha * real_data + (
            (1 - alpha) * (real_data +
                           0.5 * real_data.std() *
                           torch.rand_like(real_data))
    )
    interpolates.requires_grad_(True)
    disc_interpolates = critic(interpolates, context)
    fake = torch.ones(real_data.shape[0], 1, device=real_data.device).float().requires_grad_(True)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (gradients.norm(2, dim=1) - 1) ** 2


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
