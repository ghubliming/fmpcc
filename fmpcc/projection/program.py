import time

import numpy as np
import torch
from scipy.optimize import Bounds, minimize


class Projector:
    """DPCC projection: argmin_x' ½‖x' - x‖² subject to the constraint rows, over the normalised plan
    x (B, H, d) = [action | observation] per step, solved per candidate by SLSQP.

    Constraints (physical units, mapped into normalised coordinates):
      ('ineq', (row, d))                          row · x_t <= d on steps 1..H-1
      ('lb' | 'ub', vector)                       per channel, actions from step 0, observations from step 1
      ('sphere_outside', [ix, iy], centre, r)     keep-out disk on steps 1..H-1
      ('deriv', [x_idx, dx_idx])                  x[t+1] = x[t] + dt·dx[t]; x[0] pinned to candidate 0
    """

    def __init__(self, horizon, transition_dim, action_dim, constraints, normalizer, dt):
        self.horizon, self.transition_dim, self.action_dim = horizon, transition_dim, action_dim
        self.mins = np.concatenate([normalizer['actions'].mins, normalizer['observations'].mins])
        self.maxs = np.concatenate([normalizer['actions'].maxs, normalizer['observations'].maxs])
        self.dt = torch.tensor(dt)
        self.constraints = constraints
        self.failures = 0
        self.solve_time = 0.0
        n = transition_dim * horizon
        C, d, A, b = torch.empty((0, n)), torch.empty(0), torch.empty((0, n)), torch.empty(0)
        safety = [c for c in constraints if c[0] in ('lb', 'ub', 'eq', 'ineq')]
        self.dynamics = [c for c in constraints if c[0] == 'deriv']
        disks = [c for c in constraints if c[0] in ('sphere_inside', 'sphere_outside')]
        sC, sd, sA, sb = self._safety_rows(safety)
        dA, db = self._dynamics_rows(self.dynamics)
        C, d = torch.cat([C, sC]), torch.cat([d, sd])
        A, b = torch.cat([A, sA, dA]), torch.cat([b, sb, db])
        self.C, self.d, self.A, self.b = C.numpy(), d.numpy(), A.numpy(), b.numpy()
        self.disks = [self._disk(c) for c in disks]

    def _safety_rows(self, specs):
        H, T = self.horizon, self.transition_dim
        C, d, A, b = torch.empty((0, T * H)), torch.empty(0), torch.empty((0, T * H)), torch.empty(0)
        for spec in specs:
            kind, bound = spec[0], spec[1]
            if kind in ('lb', 'ub'):
                for dim in range(len(bound)):
                    if bound[dim] == -np.inf or bound[dim] == np.inf:
                        continue
                    mat = torch.zeros(H, T * H)
                    vec = torch.zeros(H)
                    sign = 1 if kind == 'ub' else -1
                    for t in range(H):
                        mat[t, t * T + dim] = sign
                        vec[t] = sign * bound[dim]
                    x_min, x_max = self.mins[dim], self.maxs[dim]
                    mat = mat * (x_max - x_min) / 2
                    vec = vec - sign * (x_min + x_max) / 2
                    if dim >= self.action_dim:
                        mat, vec = mat[1:], vec[1:]
                    C, d = torch.cat((C, mat), dim=0), torch.cat((d, vec), dim=0)
                continue
            mat = torch.zeros(H, T * H)
            vec = torch.zeros(H)
            for i in range(H):
                a = bound[0] * (self.maxs - self.mins) / 2
                offset = bound[1] - bound[0] @ (self.maxs + self.mins) / 2
                mat[i, i * T:(i + 1) * T] = torch.tensor(a)
                vec[i] = torch.tensor(offset)
            mat, vec = mat[1:], vec[1:]
            if kind == 'eq':
                A, b = torch.cat((A, mat), dim=0), torch.cat((b, vec), dim=0)
            else:
                C, d = torch.cat((C, mat), dim=0), torch.cat((d, vec), dim=0)
        return C, d, A, b

    def _dynamics_rows(self, specs):
        H, T = self.horizon, self.transition_dim
        A, b = torch.empty((0, T * H)), torch.empty(0)
        for spec in specs:
            x_idx, dx_idx = int(spec[1][0]), int(spec[1][1])
            mat = torch.zeros(H - 1, T * H)
            vec = torch.zeros(H - 1)
            x_diff = self.maxs[x_idx] - self.mins[x_idx]
            dx_diff = self.maxs[dx_idx] - self.mins[dx_idx]
            dx_sum = self.maxs[dx_idx] + self.mins[dx_idx]
            for i in range(H - 1):
                mat[i, i * T + x_idx] = 1 * x_diff
                mat[i, i * T + dx_idx] = self.dt * dx_diff
                mat[i, (i + 1) * T + x_idx] = -1 * x_diff
                vec[i] = - dx_sum * self.dt
            fix = torch.zeros(1, T * H)
            fix[0, x_idx] = 1
            A = torch.cat((A, torch.cat((fix, mat), dim=0)), dim=0)
            b = torch.cat((b, torch.cat((torch.tensor([0]), vec), dim=0)), dim=0)
        return A, b

    def _disk(self, spec):
        kind, dims, center, radius = spec
        T = self.transition_dim
        P, q, v = np.zeros((T, T)), np.zeros(T), radius ** 2
        for k, dim in enumerate(dims):
            delta_s = self.maxs[dim] - self.mins[dim]
            s_min = self.mins[dim]
            P[dim, dim] = delta_s ** 2 / 4
            q[dim] = delta_s ** 2 / 2 + delta_s * (s_min - center[k])
            v -= delta_s ** 2 / 4 + delta_s * (s_min - center[k]) + (s_min - center[k]) ** 2
        if kind == 'sphere_outside':
            P, q, v = -P, -q, -v
        return P, q, v

    def project(self, trajectory):
        """(B, H, d) normalised tensor -> (projected tensor on the same device, cost per candidate)."""
        start = time.perf_counter()
        H, T = self.horizon, self.transition_dim
        dims, device = trajectory.shape, trajectory.device
        batch_size = dims[0]
        flat = trajectory.reshape(batch_size, -1)
        r_np = (-flat).cpu().numpy()
        traj_np = flat.cpu().numpy()
        A, b = self.A.astype('double'), self.b.astype('double')
        C, d = self.C.astype('double'), self.d.astype('double')
        s_0 = flat[0, :T]
        for k, spec in enumerate(self.dynamics):
            b[k * H] = s_0[int(spec[1][0])]
        r_double, traj_double = r_np.astype('double'), traj_np.astype('double')
        cons = ()
        for P, q, v in self.disks:
            for t in range(1, H):
                s, e = t * T, (t + 1) * T
                cons += ({'type': 'ineq',
                          'fun': lambda x, s=s, e=e, P=P, q=q, v=v: -x[s:e] @ P @ x[s:e] - q @ x[s:e] + v,
                          'jac': lambda x, s=s, e=e, P=P, q=q: np.concatenate(
                              [np.zeros(s), -2 * P @ x[s:e] - q, np.zeros(len(x) - e)])},)
        if C.size > 0:
            cons += ({'type': 'ineq', 'fun': lambda x: -C @ x + d, 'jac': lambda x: -C},)
        if A.size > 0:
            cons += ({'type': 'eq', 'fun': lambda x: A @ x - b, 'jac': lambda x: A},)
        costs = np.ones(batch_size, dtype=np.float32)
        sol = np.zeros((batch_size, H * T), dtype=np.float32)
        box = Bounds(-5 * np.ones(H * T), 5 * np.ones(H * T))
        for i in range(batch_size):
            res = minimize(fun=lambda x: 0.5 * x @ x + r_double[i] @ x, x0=traj_double[i], constraints=cons,
                           method='SLSQP', jac=lambda x: x + r_double[i], bounds=box, tol=1e-6,
                           options={'maxiter': 1000, 'disp': False})
            self.failures += int(not res.success)
            sol[i] = res.x
            costs[i] = (0.5 * sol[i]).astype('double') @ sol[i].astype('double') + r_np[i] @ sol[i] \
                + (0.5 * traj_np[i]).astype('double') @ traj_np[i].astype('double')
        self.solve_time += time.perf_counter() - start
        return torch.tensor(sol, device=device).reshape(dims), costs
