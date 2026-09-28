import glob
import os

import torch

from .. import models
from ..data.normalizer import Normalizer


def checkpoint_path(seed_dir, which):
    if which == 'best':
        return os.path.join(seed_dir, 'checkpoint_best.pt')
    if which == 'latest':
        steps = [int(os.path.basename(p)[len('checkpoint_'):-3]) for p in glob.glob(os.path.join(seed_dir, 'checkpoint_[0-9]*.pt'))]
        if not steps:
            raise FileNotFoundError(f'no step checkpoint in {seed_dir}')
        which = max(steps)
    return os.path.join(seed_dir, f'checkpoint_{int(which):06d}.pt')


def load(seed_dir, which='best', weights='raw', device='cuda'):
    """Model (raw or moving-average weights, eval mode), normaliser and stored metadata of one checkpoint."""
    path = checkpoint_path(seed_dir, which)
    data = torch.load(path, map_location=device)
    model = models.build(data['model_name'], data['config']['objective'], data['observation_dim'], data['action_dim'],
                         data['horizon'], goal_dim=data['goal_dim'], visual=data['visual'],
                         train_steps=int(data['config']['training']['steps']))
    model.load_state_dict(data['ema' if weights == 'ema' else 'model'])
    model.to(device).eval()
    return model, Normalizer.from_state(data['normalizer']), {k: v for k, v in data.items() if k not in ('model', 'ema', 'optimizer', 'scheduler')} | {'path': path}
