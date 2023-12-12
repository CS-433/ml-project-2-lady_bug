import torch
import astropy.units as u

from src.constants import ALPHA, TIMESTEP_SCALE, N_SCALE, GAMMA_SCALE


def physics_loss_fixed_gamma_n(x, y, gamma, nH, loss_coef=1.):
    # compute dy/dx
    dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    
    # compute the physic loss
    ode = dy * TIMESTEP_SCALE - (1 - y) * gamma + ALPHA * nH * y ** 2

    return loss_coef * torch.mean(ode ** 2)

def physics_loss_varied_gamma_n(x, y, loss_coef=1.):
    # each sample in batch x should have format [timestep, gamma, nH]
    assert(x.shape[1] == 3)
    nH = x[:, 1]
    gamma = x[:, 2]
    # compute dy/dx
    dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0][:, 0]
    # compute the physic loss
    ode = dy * TIMESTEP_SCALE - (1 - y) * gamma / GAMMA_SCALE + ALPHA * nH * y ** 2 / N_SCALE

    return loss_coef * torch.mean(ode ** 2)
