import os
import pickle

import numpy as np


def load_episodes(root):
    """Quadrotor demonstrations (sorted by file name): observation [p_des, p] (6), action Δp_des (3)."""
    episodes = []
    for name in sorted(os.listdir(root)):
        if not name.endswith('.pkl'):
            continue
        with open(os.path.join(root, name), 'rb') as f:
            ep = pickle.load(f)
        actions = np.asarray(ep['actions'], dtype=np.float32)
        if len(actions) < 2:
            continue
        episodes.append({'observations': np.asarray(ep['obs'], dtype=np.float32)[:len(actions), 0:6], 'actions': actions})
    return episodes
