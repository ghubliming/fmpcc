import numpy as np


class LimitsNormalizer:
    """Maps [min, max] of the training data to [-1, 1], per dimension; a constant dimension maps to -1."""

    def __init__(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs

    @classmethod
    def fit(cls, X):
        return cls(X.min(axis=0), X.max(axis=0))

    def normalize(self, x):
        span = self.maxs - self.mins
        span[span < 1e-8] = 1.0
        x = (x - self.mins) / span
        return 2 * x - 1

    def unnormalize(self, x, eps=1e-4):
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)
        x = (x + 1) / 2.
        span = self.maxs - self.mins
        span[span < 1e-8] = 0.0
        return x * span + self.mins


class Normalizer:
    """One LimitsNormalizer per field: 'observations', 'actions'."""

    def __init__(self, fields):
        self.fields = fields

    def normalize(self, x, key):
        return self.fields[key].normalize(x)

    def unnormalize(self, x, key):
        return self.fields[key].unnormalize(x)

    def __getitem__(self, key):
        return self.fields[key]

    def state(self):
        return {k: {'mins': f.mins, 'maxs': f.maxs} for k, f in self.fields.items()}

    @classmethod
    def from_state(cls, state):
        return cls({k: LimitsNormalizer(v['mins'], v['maxs']) for k, v in state.items()})


def widen(normalizer, fraction):
    """Constant channels get the range ± fraction × the median half-range of the other channels of their field
    (at least 1e-8): the scale such a channel is mapped back with."""
    fields = {}
    for key, f in normalizer.fields.items():
        mins, maxs = f.mins.copy(), f.maxs.copy()
        const = np.asarray(mins == maxs).reshape(-1)
        if const.any():
            half = (maxs - mins)[~const] / 2.0
            half = half[np.isfinite(half) & (half > 0)]
            width = max(float(np.median(half)) * fraction, 1e-8) if half.size else 1.0
            for i in np.nonzero(const)[0]:
                mins[i] -= width
                maxs[i] += width
        fields[key] = LimitsNormalizer(mins, maxs)
    return Normalizer(fields)
