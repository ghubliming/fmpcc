import torch
import torch.nn.functional as F


def apply_conditioning(x, cond, action_dim, goal_dim=0, noise=False):
    """Pin the observation of plan step t to cond[t] (inpainting); noise=True zeroes the pinned entries.
    goal_dim > 0 also pins the last goal_dim observation channels over the whole plan to cond[0]."""
    for t, val in cond.items():
        if isinstance(t, str):
            continue
        x[:, t, action_dim:] = val.clone() if not noise else 0
    if goal_dim > 0:
        x[:, :, -goal_dim:] = cond[0][:, -goal_dim:].unsqueeze(1).clone() if not noise else 0
    return x


def loss_weights(horizon, transition_dim, action_dim, action_weight):
    weights = torch.ones(horizon, transition_dim, dtype=torch.float32)
    weights[0, :action_dim] = action_weight
    return weights


def weighted_l2(pred, targ, weights, action_dim):
    loss = F.mse_loss(pred, targ, reduction='none')
    weighted = (loss * weights).mean()
    a0 = (loss[:, 0, :action_dim] / weights[0, :action_dim]).mean()
    return weighted, {'loss': weighted, 'a0_loss': a0}


def flow_gate(k, K, threshold):
    """Flow samplers project on step k of K when k >= int((1 - threshold) K), and always on the last."""
    return k >= int((1.0 - threshold) * K) or k == K - 1
