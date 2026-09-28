import math

import torch
from torch import nn
from torch.func import jvp

from .common import apply_conditioning, flow_gate


class AnalyticMeanFM(nn.Module):
    """Analytic-MeanFM: average-velocity flow (MeanFlow, Geng et al., 2025), u(z_r, r, h) over [r, r + h], noise at
    τ = 0. Analytic target u = v + h du/dr by a forward-mode JVP with tangents (v, 1, -1); logit-normal time pair; a
    share of the batch at h = 0; adaptive L2 err / sg(err + eps)^p on the per-sample sum; a second head
    regresses the instantaneous velocity. Sampling: z += u(z, k/K, 1/K) / K from unit noise."""

    noise_scale = 1.0
    two_time = True

    def __init__(self, network, horizon, observation_dim, action_dim, goal_dim=0, encoder=None,
                 time_logit_normal=(-0.4, 1.0), fm_share=0.5, adaptive_p=1.0, adaptive_eps=0.01):
        super().__init__()
        self.network = network
        self.encoder = encoder
        self.horizon, self.observation_dim, self.action_dim = horizon, observation_dim, action_dim
        self.transition_dim = observation_dim + action_dim
        self.goal_dim = goal_dim
        self.p_mean, self.p_std = float(time_logit_normal[0]), float(time_logit_normal[1])
        self.fm_share = float(fm_share)
        self.adaptive_p, self.adaptive_eps = float(adaptive_p), float(adaptive_eps)

    def prepare(self, cond):
        """Encode the images once per plan (the latent is constant over the steps and inside the JVP)."""
        if self.encoder is None or 'visual' not in cond:
            return cond
        out = {k: v for k, v in cond.items() if k != 'visual'}
        out['visual_latent'] = self.encoder(*cond['visual'])
        return out

    def uv(self, x, cond, t, h):
        return self.network(x, t, h=h, cond=cond.get('visual_latent'), return_v=True)

    def velocity(self, x, cond, t):
        """Instantaneous field u(x, t, h = 0) for the endpoint sampler."""
        return self.uv(x, cond, t, torch.zeros_like(t))[0]

    def _time_pair(self, B, device):
        taus = torch.sigmoid(torch.randn(2, B, device=device) * self.p_std - self.p_mean)
        t = torch.maximum(taus[0], taus[1])
        r = torch.minimum(taus[0], taus[1])
        fm_mask = torch.rand(B, device=device) < self.fm_share
        r = torch.where(fm_mask, t, r)
        return r, t - r

    def _anchor(self, x, cond):
        ad, gd = self.action_dim, self.goal_dim
        r, h = self._time_pair(x.shape[0], x.device)
        x_base = apply_conditioning(torch.randn_like(x), cond, ad, gd, noise=True)
        r_exp = r[:, None, None]
        x_r = apply_conditioning((1.0 - r_exp) * x_base + r_exp * x, cond, ad, gd)
        v_inst = apply_conditioning(x - x_base, cond, ad, gd, noise=True)
        return r, h, x_r, v_inst

    def _jvp_target(self, x_r, r, h, v_inst, cond):
        u_of = lambda z, r_in, h_in: self.uv(z, cond, r_in, h_in)[0]
        ones = torch.ones_like(r)
        u, du_dr = jvp(u_of, (x_r, r, h), (v_inst, ones, -ones))
        return u, (v_inst + h[:, None, None] * du_dr).detach()

    def _adaptive(self, err):
        return err / (err + self.adaptive_eps).pow(self.adaptive_p).detach()

    def loss(self, x, cond):
        cond = self.prepare(cond)
        r, h, x_r, v_inst = self._anchor(x, cond)
        u_pred, u_target = self._jvp_target(x_r, r, h, v_inst, cond)
        u_target = apply_conditioning(u_target, cond, self.action_dim, self.goal_dim, noise=True)
        v_pred = self.uv(x_r, cond, r, h)[1]
        err_u = (u_pred - u_target).pow(2).sum(dim=(1, 2))
        err_v = (v_pred - v_inst.detach()).pow(2).sum(dim=(1, 2))
        loss = (self._adaptive(err_u) + self._adaptive(err_v)).mean()
        return loss, {'loss': loss.detach(), 'mse_u': err_u.detach().mean(), 'mse_v': err_v.detach().mean()}

    @torch.no_grad()
    def sample(self, cond, batch_size, steps, projector=None, threshold=0.5, goal_dim=None):
        goal_dim = self.goal_dim if goal_dim is None else goal_dim
        cond = self.prepare(cond)
        device = next(self.network.parameters()).device
        x = torch.randn((batch_size, self.horizon, self.transition_dim), device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim)
        dt = 1.0 / steps
        h = torch.full((batch_size,), dt, device=device, dtype=torch.float32)
        costs = {}
        for k in range(steps):
            t = torch.full((batch_size,), k / steps, device=device, dtype=torch.float32)
            x = x + self.uv(x, cond, t, h)[0] * dt
            x = apply_conditioning(x, cond, self.action_dim, goal_dim)
            if projector is not None:
                if flow_gate(k, steps, threshold):
                    x, costs[k] = projector.project(x)
                x = apply_conditioning(x, cond, self.action_dim, goal_dim)
        return x, costs


class CIMeanFM(AnalyticMeanFM):
    """CI-MeanFM (α-Flow, Zhang et al., 2025): the target interpolates from flow matching (α = 1) to the
    Analytic-MeanFM target (α = 0) over training, α on a sigmoid schedule; for 0 < α < 1 the bootstrap target
    (α h v + (1 - α) h u(z + α h v, r + α h, (1 - α) h)) / h, clamped, replaces the JVP."""

    def __init__(self, *args, alpha_end=0.0, alpha_gamma=25.0, alpha_clamp=0.005, alpha_end_step=100000,
                 target_clamp=4.0, adaptive_eps=1e-3, **kwargs):
        super().__init__(*args, adaptive_eps=adaptive_eps, **kwargs)
        self.alpha_init, self.alpha_end = 1.0, float(alpha_end)
        self.alpha_gamma, self.alpha_clamp = float(alpha_gamma), float(alpha_clamp)
        self.alpha_end_step = int(alpha_end_step)
        self.target_clamp = float(target_clamp)
        self.train_step = 0

    def alpha(self):
        middle = self.alpha_end_step / 2.0
        progress = (self.train_step - middle) / max(self.alpha_end_step, 1)
        if self.train_step > self.alpha_end_step:
            ratio = self.alpha_end
        else:
            ratio = self.alpha_init + (self.alpha_end - self.alpha_init) * (1.0 / (1.0 + math.exp(-progress * self.alpha_gamma)))
        if ratio < self.alpha_clamp:
            return 0.0
        if ratio > 1.0 - self.alpha_clamp:
            return 1.0
        return float(ratio)

    def _adaptive(self, err):
        return err / (err + self.adaptive_eps).detach()

    def loss(self, x, cond):
        cond = self.prepare(cond)
        alpha = self.alpha()
        r, h, x_r, v_inst = self._anchor(x, cond)
        h_exp = h[:, None, None]
        if alpha <= 0.0:
            u_target = self._jvp_target(x_r, r, h, v_inst, cond)[1]
        elif alpha >= 1.0:
            u_target = v_inst.clone()
        else:
            dt = alpha * h
            dt_exp = dt[:, None, None]
            with torch.no_grad():
                u_next = self.uv(x_r + dt_exp * v_inst, cond, r + dt, h - dt)[0]
                u_target = (dt_exp * v_inst + (h_exp - dt_exp) * u_next) / h_exp.clamp(min=1e-12)
                u_target = u_target.clamp(-self.target_clamp, self.target_clamp)
                u_target = torch.where(h_exp > 0, u_target, v_inst)
        u_target = apply_conditioning(u_target, cond, self.action_dim, self.goal_dim, noise=True).detach()
        u_pred, v_pred = self.uv(x_r, cond, r, h)
        err_u = (u_pred - u_target).pow(2).sum(dim=(1, 2))
        err_v = (v_pred - v_inst.detach()).pow(2).sum(dim=(1, 2))
        branch = (h > 0) & (alpha > 0.0)
        w = torch.where(branch, torch.full_like(err_u, alpha), torch.ones_like(err_u))
        loss = (w * self._adaptive(err_u) + self._adaptive(err_v)).mean()
        return loss, {'loss': loss.detach(), 'mse_u': err_u.detach().mean(), 'mse_v': err_v.detach().mean(),
                      'alpha': torch.tensor(alpha)}
