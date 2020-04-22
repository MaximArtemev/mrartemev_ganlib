"""
Implements various gans.

Reference:

"""

import logging
import torch
from torch import nn

from .gan_utils import dragan_gradient_penalty
from .gans import JSGAN, WGAN

logger = logging.getLogger('main.ganlib.gans.dragan')


class DRAGAN_W(WGAN):
    """
    ToDo add explanation
    Alert: here GP uses a different gen sample that disc_loss
    """

    def __init__(self, dim, prior, base_network, **base_network_kwargs):
        super().__init__(dim, prior, base_network, **base_network_kwargs)

    def calculate_loss_disc(self, input, context=None, lambda_gp=10):
        d_loss = super().calculate_loss_disc(input, context)
        gp = dragan_gradient_penalty(self.discriminator,
                                     input.data,
                                     context)
        d_loss = d_loss + lambda_gp * gp
        return d_loss


class DRAGAN_JS(JSGAN):
    """
    ToDo add explanation
    Alert: here GP uses a different gen sample that disc_loss
    """

    def __init__(self, dim, prior, base_network, **base_network_kwargs):
        super().__init__(dim, prior, base_network, **base_network_kwargs)

    def calculate_loss_disc(self, input, context=None, lambda_gp=10):
        d_loss = super().calculate_loss_disc(input, context)
        gp = dragan_gradient_penalty(self.discriminator,
                                     input.data,
                                     context)
        d_loss = d_loss + lambda_gp * gp
        return d_loss
