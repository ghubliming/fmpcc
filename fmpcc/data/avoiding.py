import os
import pickle

import numpy as np


def load_episodes(root):
    """D3IL obstacle-avoidance demonstrations: observation [x_des, y_des, x, y], action Δ[x_des, y_des]."""
    episodes = []
    for name in sorted(os.listdir(root)):
        with open(os.path.join(root, name), 'rb') as f:
            state = pickle.load(f)
        des = state['robot']['des_c_pos'][:, :2]
        pos = state['robot']['c_pos'][:, :2]
        episodes.append({
            'observations': np.concatenate((des, pos), axis=-1)[:-1],
            'actions': des[1:] - des[:-1],
        })
    return episodes
