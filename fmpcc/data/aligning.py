import glob
import os
import pickle

import cv2
import numpy as np
import torch

from .normalizer import LimitsNormalizer, Normalizer

SPLITS = os.path.join(os.path.dirname(__file__), '..', 'envs', 'd3il', 'data', 'aligning')


def _frames(root, camera, name):
    paths = sorted(glob.glob(os.path.join(root, 'images', camera, name, '*')),
                   key=lambda p: int(os.path.basename(p).split('.')[0]))
    return np.stack([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]) if paths \
        else np.zeros((0, 96, 96, 3), dtype=np.uint8)


def _image(frame):
    return torch.from_numpy((frame.astype(np.float32) / 255.0).transpose(2, 0, 1))[None]


class AligningWindows(torch.utils.data.Dataset):
    """D3IL alignment demonstrations: plan windows [Δdes (3) | des (3) | pos (3)] of the end effector,
    conditioned on the first step's state and its two camera frames (RGB, [0, 1])."""

    def __init__(self, root, horizon, files=os.path.join(SPLITS, 'train_files.pkl'), max_episodes=1000):
        self.horizon = horizon
        names = np.load(files, allow_pickle=True)[:max_episodes]
        self.obs, self.act, self.agentview, self.in_hand = [], [], [], []
        for file in names:
            with open(os.path.join(root, 'state', file), 'rb') as f:
                state = pickle.load(f)
            des, pos = state['robot']['des_c_pos'], state['robot']['c_pos']
            T = len(des) - 1
            self.obs.append(np.concatenate([des[:T], pos[:T]], axis=-1).astype(np.float32))
            self.act.append((des[1:] - des[:-1]).astype(np.float32))
        self.normalizer = Normalizer({'observations': LimitsNormalizer.fit(np.concatenate(self.obs)),
                                      'actions': LimitsNormalizer.fit(np.concatenate(self.act))})
        for file in names:
            name = os.path.basename(file).split('.')[0]
            self.agentview.append(_frames(root, 'bp-cam', name))
            self.in_hand.append(_frames(root, 'inhand-cam', name))
        self.indices = np.array([(ep, s, s + horizon) for ep in range(len(names))
                                 for s in range(min(len(self.obs[ep]), len(self.agentview[ep])) - horizon + 1)],
                                dtype=np.int64)
        self.observation_dim, self.action_dim, self.goal_dim = 6, 3, 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep, start, end = self.indices[idx]
        obs = self.normalizer.normalize(self.obs[ep][start:end], 'observations').astype(np.float32)
        act = self.normalizer.normalize(self.act[ep][start:end], 'actions').astype(np.float32)
        cond = {0: obs[0], 'visual': (_image(self.agentview[ep][start]), _image(self.in_hand[ep][start]))}
        return np.concatenate([act, obs], axis=-1), cond
