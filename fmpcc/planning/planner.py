import time

import numpy as np
import torch

from ..projection.endpoint import EndpointSampler


def guiding_steps(steps, threshold):
    """Guiding (active, non-terminal) steps of endpoint projection at K = steps."""
    return max(steps - int((1.0 - threshold) * steps), 1) - 1


def skip_reason(model, projection, steps, threshold):
    """Why a variant cannot run: endpoint projection needs a velocity field and at least one guiding step."""
    if projection != 'endpoint':
        return None
    if not hasattr(model, 'velocity'):
        return 'endpoint projection needs a velocity field (flow models only)'
    if guiding_steps(steps, threshold) == 0:
        return f'endpoint projection has no guiding step at K={steps}, eta={threshold}'
    return None


class Planner:
    """One receding-horizon planning call: B candidate plans from independent noise, sampled unguided,
    with per-step projection (DPCC) or endpoint projection, then one candidate selected.

    rule: 'random' (the first candidate), 'cumulative_cost' (least total projection cost) or
    'temporal_consistency' (closest to the previously executed plan: shifted observations, or the full
    normalised plan unshifted when consistency='full'). unnormalize='selected' maps only the executed
    plan's actions back (the clip gate then sees that plan alone).
    """

    def __init__(self, model, normalizer, projection, rule, candidates, steps, threshold, projector=None,
                 goal_dim=None, consistency='shifted', unnormalize='batch'):
        self.model, self.normalizer = model, normalizer
        self.projection, self.rule, self.candidates = projection, rule, candidates
        self.steps, self.threshold = steps, threshold
        self.projector = projector
        self.goal_dim = goal_dim
        self.consistency = consistency
        self.unnormalize = unnormalize
        self.endpoint = EndpointSampler(model, projector, threshold) if projection == 'endpoint' else None
        self.previous = None
        self.device = next(model.parameters()).device

    def condition(self, observation, extra=None):
        obs = torch.tensor(self.normalizer.normalize(observation, 'observations'), dtype=torch.float32)
        cond = {0: obs.unsqueeze(0).repeat(self.candidates, 1).to(self.device)}
        if extra:
            cond.update(extra)
        return cond

    def __call__(self, observation, extra=None):
        """Returns the first action of the selected plan and the plan record."""
        B = self.candidates
        cond = self.condition(observation, extra)
        solved = self.projector.solve_time if self.projector is not None else 0.0
        start = time.time()
        if self.projection == 'endpoint':
            x, cost = self.endpoint.sample(cond, B, self.steps)
            costs = {0: cost}
        else:
            projector = self.projector if self.projection == 'per_step' else None
            x, costs = self.model.sample(cond, B, self.steps, projector=projector, threshold=self.threshold,
                                         goal_dim=self.goal_dim)
        plans = x.cpu().numpy()
        elapsed = time.time() - start
        projection_time = (self.projector.solve_time - solved) if self.projector is not None else 0.0
        ad = self.model.action_dim
        observations = self.normalizer.unnormalize(plans[:, :, ad:], 'observations')
        which = self.select(plans, observations, costs)
        self.previous = (plans[which], observations[which])
        if self.unnormalize == 'selected':
            action = self.normalizer.unnormalize(plans[which, :, :ad], 'actions')[0]
        else:
            action = self.normalizer.unnormalize(plans[:, :, :ad], 'actions')[which, 0]
        return action, {'observations': observations, 'normed': plans, 'executed': int(which), 'time': elapsed,
                        'projection_time': projection_time if self.projection != 'unguided' else 0.0}

    def set_projector(self, projector):
        self.projector = projector
        if self.endpoint is not None:
            self.endpoint.projector = projector

    def select(self, plans, observations, costs):
        if self.projection == 'unguided' or self.candidates == 1:
            return 0
        if self.rule == 'cumulative_cost':
            total = np.zeros(self.candidates)
            for c in costs.values():
                total += c
            return int(np.argmin(total))
        if self.rule == 'temporal_consistency' and self.previous is not None:
            if self.consistency == 'full':
                return int(np.argsort(np.linalg.norm(plans - self.previous[0][None], axis=(1, 2)))[0])
            prev = self.previous[1][None]
            return int(np.argsort(np.linalg.norm(observations[:, :-1, :] - prev[:, 1:, :], axis=(1, 2)))[0])
        return 0
