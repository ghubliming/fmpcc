import torch
from torch import nn

from .common import apply_conditioning, flow_gate, loss_weights, weighted_l2


class FlowMatching(nn.Module):
    """Instantaneous-velocity flow matching (Lipman et al., 2023) on plans, noise at τ = 0, data at τ = 1.
    Training time τ = 1 - Beta(a, b); explicit Euler with K steps; initial noise scaled by 0.5 (the
    sampler inherited from DPCC)."""

    noise_scale = 0.5
    two_time = False

    def __init__(self, network, horizon, observation_dim, action_dim, time_beta=(1.5, 1.0), action_weight=10.,
                 goal_dim=0, encoder=None):
        super().__init__()
        self.network = network
        self.encoder = encoder
        self.horizon, self.observation_dim, self.action_dim = horizon, observation_dim, action_dim
        self.transition_dim = observation_dim + action_dim
        self.goal_dim = goal_dim
        self.time_beta = (float(time_beta[0]), float(time_beta[1]))
        self.register_buffer('loss_weights', loss_weights(horizon, self.transition_dim, action_dim, action_weight))

    def prepare(self, cond):
        return cond

    def velocity(self, x, cond, t):
        feature = self.encoder(*cond['visual']) if self.encoder is not None else None
        return self.network(x, t, cond=feature)

    def loss(self, x, cond):
        a = torch.tensor(self.time_beta[0], device=x.device)
        b = torch.tensor(self.time_beta[1], device=x.device)
        t = 1.0 - torch.distributions.Beta(a, b).sample((x.shape[0],))
        x_base = apply_conditioning(torch.randn_like(x), cond, self.action_dim, self.goal_dim, noise=True)
        t_exp = t[:, None, None]
        x_t = (1.0 - t_exp) * x_base + t_exp * x
        x_t = apply_conditioning(x_t, cond, self.action_dim, self.goal_dim)
        v_target = apply_conditioning(x - x_base, cond, self.action_dim, self.goal_dim, noise=True)
        return weighted_l2(self.velocity(x_t, cond, t), v_target, self.loss_weights, self.action_dim)

    @torch.no_grad()
    def sample(self, cond, batch_size, steps, projector=None, threshold=0.5, goal_dim=None):
        goal_dim = self.goal_dim if goal_dim is None else goal_dim
        device = self.loss_weights.device
        x = 0.5 * torch.randn((batch_size, self.horizon, self.transition_dim), device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim)
        dt = 1.0 / steps
        costs = {}
        for k in range(steps):
            t = torch.full((batch_size,), k / steps, device=device, dtype=torch.float32)
            x = x + self.velocity(x, cond, t) * dt
            x = apply_conditioning(x, cond, self.action_dim, goal_dim)
            if projector is not None and flow_gate(k, steps, threshold):
                x, costs[k] = projector.project(x)
            x = apply_conditioning(x, cond, self.action_dim, goal_dim)
        return x, costs
