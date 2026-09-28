import numpy as np
import torch

from ..models.common import apply_conditioning, flow_gate


class EndpointSampler:
    """Endpoint projection (HardFlow, Li et al., 2025, Alg. 1 with cost C = 0) on an Euler flow sampler.

    On every active step the endpoint predicted from the lookahead point is projected with the DPCC
    program and the state is pulled back by τ_{k+1} (Π(x̂₁) - x̂₁). Decision variables drop the pinned
    initial state s₀. The two-time models are queried at h = 0.
    """

    def __init__(self, model, projector, threshold):
        self.model, self.projector, self.threshold = model, projector, threshold
        self.H, self.T, self.ad = projector.horizon, projector.transition_dim, projector.action_dim
        self.nfe = 0

    def _from_dof(self, dof, s0):
        return np.insert(np.asarray(dof).reshape(-1), self.ad, np.asarray(s0).reshape(-1))

    def _to_dof(self, flat):
        return np.concatenate([flat[..., :self.ad], flat[..., self.T:]], axis=-1)

    def _velocity(self, X, tau, s0, cond):
        B = X.shape[0]
        traj = torch.cat([X[:, :self.ad], s0, X[:, self.ad:]], dim=1).view(B, self.H, self.T)
        t = torch.full((B,), float(tau), dtype=torch.float32, device=X.device)
        with torch.no_grad():
            v = self.model.velocity(traj, cond, t)
        self.nfe += B
        v = v.reshape(B, -1)
        return torch.cat([v[:, :self.ad], v[:, self.T:]], dim=1)

    def _project(self, x1_ref, s0):
        # The thesis runs projected with float64 normaliser limits; float32 here, so paths may differ very slightly.
        full = self._from_dof(np.asarray(x1_ref, dtype=float), np.asarray(s0, dtype=float))
        sol, _ = self.projector.project(torch.tensor(full.reshape(1, self.H, self.T), dtype=torch.float32))
        return self._to_dof(np.asarray(sol.numpy(), dtype=float).reshape(-1))

    @torch.no_grad()
    def sample(self, cond, batch_size, steps):
        """Returns the plans (B, H, d) and the per-candidate cost Σ_k ‖Π(x̂₁) - x̂₁‖²."""
        cond = self.model.prepare(cond)
        device = next(self.model.parameters()).device
        K, dt = int(steps), 1.0 / steps
        x_init = self.model.noise_scale * torch.randn(batch_size, self.H, self.T, device=device)
        x_init = apply_conditioning(x_init, cond, self.ad)
        flat = x_init.reshape(batch_size, -1)
        X = torch.cat([flat[:, :self.ad], flat[:, self.T:]], dim=1)
        s0 = x_init[:, 0, self.ad:]
        s0_np = s0.cpu().numpy()
        cost = np.zeros(batch_size, dtype=np.float64)
        for k in range(K):
            tau_k = k * dt
            tau_next = tau_k + dt
            X_ref = X + self._velocity(X, tau_k, s0, cond) * dt
            if flow_gate(k, K, self.threshold):
                if k < K - 1:
                    X1_ref = X_ref + (1.0 - tau_next) * self._velocity(X_ref, tau_next, s0, cond)
                else:
                    X1_ref = X_ref
                X1_ref_np = X1_ref.cpu().numpy()
                X1_proj_np = np.empty_like(X1_ref_np)
                for b in range(batch_size):
                    X1_proj_np[b] = np.asarray(self._project(X1_ref_np[b], s0_np[b]), dtype=X1_ref_np.dtype)
                cost += np.sum((X1_proj_np.astype(np.float64) - X1_ref_np.astype(np.float64)) ** 2, axis=1)
                X1_proj = torch.as_tensor(X1_proj_np, dtype=X_ref.dtype, device=device)
                X = X_ref + tau_next * (X1_proj - X1_ref)
            else:
                X = X_ref
        X_np = X.cpu().numpy()
        out = torch.zeros_like(x_init)
        for b in range(batch_size):
            out[b] = torch.as_tensor(self._from_dof(X_np[b], s0_np[b]), dtype=torch.float32, device=device).view(self.H, self.T)
        return apply_conditioning(out, cond, self.ad), cost
