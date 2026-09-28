import os

import numpy as np

SPLITS = os.path.join(os.path.dirname(__file__), 'd3il', 'data', 'aligning')
TABLE = (np.array([-0.30, -1.20, -0.50]), np.array([1.60, 1.20, 1.50]))
ROUTE = (np.array([0.20, -0.45, 0.02]), np.array([0.80, 0.45, 0.50]))
ROUTE_SLACK = 0.15
BOX_HALF_SIDE = 0.05
BOX_MAX_OVERLAP = 0.005


def make(max_steps=400):
    from .d3il.envs.gym_aligning.envs.aligning import Robot_Push_Env
    return Robot_Push_Env(render=False, if_vision=True, max_steps_per_episode=max_steps)


def contexts(split='train'):
    return np.load(os.path.join(SPLITS, f'{split}_contexts.pkl'), allow_pickle=True)


def box_blocked(context, disks, enlarge):
    """True if a keep-out disk (enlarged) overlaps the initial box footprint by more than 5 mm: the end
    effector cannot reach the box without entering the disk, so the rollout holds position."""
    pos = context[0]
    th = np.deg2rad(float(pos[2]))
    c, s = np.cos(th), np.sin(th)
    for disk in disks:
        d = np.asarray(disk['center'][:2], dtype=float) - np.asarray(pos[:2], dtype=float)
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        sep = float(np.linalg.norm(local - np.clip(local, -BOX_HALF_SIDE, BOX_HALF_SIDE)))
        if max(0.0, float(disk['radius']) + enlarge - sep) > BOX_MAX_OVERLAP:
            return True
    return False


def diverged(des, pos):
    """Reason the end effector is lost (non-finite, off the table, off the task envelope ± 0.15 m) or None."""
    des, pos = np.asarray(des, dtype=float)[:3], np.asarray(pos, dtype=float)[:3]
    if not (np.all(np.isfinite(des)) and np.all(np.isfinite(pos))):
        return 'nan_state'
    if np.any(pos < TABLE[0]) or np.any(pos > TABLE[1]):
        return 'off_table'
    if np.any(pos < ROUTE[0] - ROUTE_SLACK) or np.any(pos > ROUTE[1] + ROUTE_SLACK):
        return 'off_route'
    return None


def images(bp, inhand):
    """Camera frames (H, W, 3) uint8 -> (3, H, W) float in [0, 1]."""
    return bp.transpose((2, 0, 1)).copy() / 255., inhand.transpose((2, 0, 1)).copy() / 255.
