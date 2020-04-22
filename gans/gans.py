"""
Implements various gans.

Reference:

"""

import logging
import torch
from torch import nn

from .gan_utils import gradient_penalty

logger = logging.getLogger('main.ganlib.gans.jsgan')


class GAN(nn.Module):
    """
    Default GAN module
    """

    def __init__(self, dim, prior, base_network, **base_network_kwargs):
        super().__init__()
        self.prior = prior
        # input shape = prior.event_shape[0]
        self.generator = base_network(prior.event_shape[0], dim, **base_network_kwargs)
        self.discriminator = base_network(dim, 1, **base_network_kwargs)
        self.register_buffer('placeholder', torch.randn(1))

    def generate(self, num_samples, context=None):
        z = self.prior.sample((num_samples,)).to(self.placeholder.device)
        return self.generator(z, context)

    def discriminate(self, input, context=None):
        return self.discriminator(input, context)

    def calculate_loss_gen(self, input, context=None):
        raise Exception("Not Implemented")

    def calculate_loss_disc(self, input, context=None):
        raise Exception("Not Implemented")

    def forward(self, *args, **kwargs):
        raise Exception("Not Implemented")


class JSGAN(GAN):
    """
    ToDo add explanation
    """

    def __init__(self, dim, prior, base_network, **base_network_kwargs):
        super().__init__(dim, prior, base_network, **base_network_kwargs)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def calculate_loss_disc(self, input, context=None):
        generated = self.generate(input.shape[0], context)
        discriminated_real = self.discriminate(input, context)
        discriminated_fake = self.discriminate(generated.detach(), context)
        discriminator_loss = 0.5 * (self.criterion(discriminated_real, torch.ones_like(discriminated_real)) +
                                    self.criterion(discriminated_fake, torch.zeros_like(discriminated_fake)))
        return discriminator_loss.view(-1, 1)

    def calculate_loss_gen(self, input, context=None):
        generated = self.generate(input.shape[0], context)
        discriminated_fake = self.discriminate(generated, context)
        generator_loss = self.criterion(discriminated_fake, torch.ones_like(discriminated_fake))
        return generator_loss.view(-1, 1)


class LSGAN(JSGAN):
    """
    ToDo add explanation
    """

    def __init__(self, dim, prior, base_network, **base_network_kwargs):
        super().__init__(dim, prior, base_network, **base_network_kwargs)
        self.discriminator.add_module("Sigmoid", nn.Sigmoid())
        self.criterion = nn.MSELoss(reduction='none')


class WGAN(GAN):
    """
    ToDo add explanation
    """

    def __init__(self, dim, prior, base_network, **base_network_kwargs):
        super().__init__(dim, prior, base_network, **base_network_kwargs)

    def calculate_loss_disc(self, input, context=None, lambda_gp=10):
        self.generator.eval()
        self.discriminator.train()
        generated = self.generate(input.shape[0], context)
        discriminated_real = self.discriminate(input, context)
        discriminated_fake = self.discriminate(generated, context)
        d_loss = - discriminated_real + \
                 discriminated_fake
        return d_loss.view(-1, 1)

    def calculate_loss_gen(self, input, context=None):
        generated = self.generate(input.shape[0], context)
        discriminated_fake = self.discriminate(generated, context)
        g_loss = - discriminated_fake
        return g_loss.view(-1, 1)


class WGAN_GP(WGAN):
    """
    ToDo add explanation
    Alert: here GP uses a different gen sample that disc_loss
    """

    def __init__(self, dim, prior, base_network, **base_network_kwargs):
        super().__init__(dim, prior, base_network, **base_network_kwargs)

    def calculate_loss_disc(self, input, context=None, lambda_gp=10):
        d_loss = super().calculate_loss_disc(input, context, lambda_gp)
        gp = gradient_penalty(self.discriminator,
                              input.data,
                              self.generate(input.shape[0], context).data,
                              context)
        # d_loss = d_loss + lambda_gp * gp
        return (d_loss + lambda_gp * gp).view(-1, 1)


class CramerGAN(GAN):
    """
    ToDo add explanation
    """

    def __init__(self, dim, prior, base_network, cramer_dim=64, **base_network_kwargs):
        super().__init__(dim, prior, base_network, **base_network_kwargs)
        self.discriminator = base_network(dim, cramer_dim, **base_network_kwargs)

    def cramer_surrogate(self, input_1, input_2, context):
        discriminated_1 = self.discriminate(input_1, context)
        discriminated_2 = self.discriminate(input_2, context)
        return torch.norm(discriminated_1 - discriminated_2, dim=1) - \
               torch.norm(discriminated_1, dim=1)

    def calculate_loss_disc(self, input, context=None, lambda_gp=10):
        # surrogate
        generated_1 = self.generate(input.shape[0], context)
        generated_2 = self.generate(input.shape[0], context)
        surrogate = self.cramer_surrogate(input, generated_1, context) - \
                    self.cramer_surrogate(generated_1, generated_2, context)
        gp = gradient_penalty(self.discriminator,
                              input.data,
                              generated_1.data,
                              context)
        # return lambda_gp * gp - surrogate
        return (- surrogate + lambda_gp * gp).view(-1, 1)

    def calculate_loss_gen(self, input, context=None):
        # real (not surrogate)
        discriminated_1 = self.discriminate(self.generate(input.shape[0], context), context)
        discriminated_2 = self.discriminate(self.generate(input.shape[0], context), context)
        discriminated_real = self.discriminate(input, context)
        gen_loss = torch.norm(discriminated_real - discriminated_1, dim=1) + \
                   torch.norm(discriminated_real - discriminated_2, dim=1) - \
                   torch.norm(discriminated_1 - discriminated_2, dim=1)
        return gen_loss.view(-1, 1)
