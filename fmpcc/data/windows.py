import numpy as np
import torch

from .normalizer import LimitsNormalizer, Normalizer, widen


class PlanWindows(torch.utils.data.Dataset):
    """Fixed-length plan windows [action | observation] of padded demonstrations (DPCC dataset layout).

    episodes: list of dicts with 'observations' (T, obs_dim) and 'actions' (T, act_dim).
    """

    def __init__(self, episodes, horizon, max_path_length, use_padding=True, constant_margin=None):
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        n = len(episodes)
        obs_dim = episodes[0]['observations'].shape[-1]
        act_dim = episodes[0]['actions'].shape[-1]
        self.observations = np.zeros((n, max_path_length, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n, max_path_length, act_dim), dtype=np.float32)
        self.path_lengths = np.zeros(n, dtype=int)
        for i, ep in enumerate(episodes):
            L = min(len(ep['observations']), max_path_length)
            self.observations[i, :L] = ep['observations'][:L]
            self.actions[i, :L] = ep['actions'][:L]
            self.path_lengths[i] = L
            if use_padding:
                self.observations[i, L:] = self.observations[i, L - 1]
                self.actions[i, L:] = self.actions[i, L - 1]
        self.raw_normalizer = Normalizer({
            'observations': LimitsNormalizer.fit(self._flat(self.observations)),
            'actions': LimitsNormalizer.fit(self._flat(self.actions)),
        })
        normalizer = self.raw_normalizer if constant_margin is None else widen(self.raw_normalizer, constant_margin)
        self.normalizer = normalizer
        self.observation_dim, self.action_dim = obs_dim, act_dim
        first = int(np.argmax(self.path_lengths > 1))
        self.goal_dim = int((self.observations[first].std(axis=0) == 0).sum())
        self.normed_observations = normalizer.normalize(
            self.observations.reshape(n * max_path_length, -1), 'observations').reshape(n, max_path_length, -1)
        self.normed_actions = normalizer.normalize(
            self.actions.reshape(n * max_path_length, -1), 'actions').reshape(n, max_path_length, -1)
        self.indices = self._indices()

    def _flat(self, x):
        return np.concatenate([x[i, :L] for i, L in enumerate(self.path_lengths)], axis=0)

    def _indices(self):
        indices = []
        for i, L in enumerate(self.path_lengths):
            max_start = min(L - 1, self.max_path_length - self.horizon)
            if not self.use_padding:
                max_start = min(max_start, L - self.horizon)
            indices.extend((i, s, s + self.horizon) for s in range(max_start))
        return np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, start, end = self.indices[idx]
        observations = self.normed_observations[i, start:end]
        actions = self.normed_actions[i, start:end]
        return np.concatenate([actions, observations], axis=-1), {0: observations[0]}
