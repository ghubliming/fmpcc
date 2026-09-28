import numpy as np
import torch
from torch import nn

from .common import apply_conditioning, loss_weights, weighted_l2


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, a_min=0, a_max=0.999), dtype=dtype)


def extract(a, t, x_shape):
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):
    """DDPM of Diffuser/DPCC: cosine schedule, epsilon prediction, K reverse steps fixed at training.
    Initial and reverse-step noise scaled by 0.5 as in DPCC."""

    noise_scale = 0.5
    two_time = False

    def __init__(self, network, horizon, observation_dim, action_dim, steps=20, action_weight=10.,
                 goal_dim=0, encoder=None):
        super().__init__()
        self.network = network
        self.encoder = encoder
        self.horizon, self.observation_dim, self.action_dim = horizon, observation_dim, action_dim
        self.transition_dim = observation_dim + action_dim
        self.goal_dim = goal_dim
        self.n_timesteps = int(steps)
        betas = cosine_beta_schedule(self.n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.register_buffer('loss_weights', loss_weights(horizon, self.transition_dim, action_dim, action_weight))

    def prepare(self, cond):
        return cond

    def _net(self, x, t, cond):
        feature = self.encoder(*cond['visual']) if self.encoder is not None else None
        return self.network(x, t, cond=feature)

    def loss(self, x, cond):
        t = torch.randint(0, self.n_timesteps, (x.shape[0],), device=x.device).long()
        noise = apply_conditioning(torch.randn_like(x), cond, self.action_dim, self.goal_dim, noise=True)
        x_noisy = extract(self.sqrt_alphas_cumprod, t, x.shape) * x + \
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim, self.goal_dim)
        return weighted_l2(self._net(x_noisy, t, cond), noise, self.loss_weights, self.action_dim)

    @torch.no_grad()
    def _p_sample(self, x, cond, t):
        epsilon = self._net(x, t, cond)
        x_recon = extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * epsilon
        mean = extract(self.posterior_mean_coef1, t, x.shape) * x_recon + \
            extract(self.posterior_mean_coef2, t, x.shape) * x
        log_var = extract(self.posterior_log_variance_clipped, t, x.shape)
        noise = 0.5 * torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return mean + nonzero_mask * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def sample(self, cond, batch_size, steps=None, projector=None, threshold=0.5, goal_dim=None):
        """Reverse chain; the projector acts after every step t <= threshold * K. steps is fixed at training."""
        goal_dim = self.goal_dim if goal_dim is None else goal_dim
        device = self.betas.device
        x = 0.5 * torch.randn((batch_size, self.horizon, self.transition_dim), device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim)
        costs = {}
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self._p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim, goal_dim)
            if projector is not None and i <= threshold * self.n_timesteps:
                x, costs[i] = projector.project(x)
            x = apply_conditioning(x, cond, self.action_dim, goal_dim)
        return x, costs
