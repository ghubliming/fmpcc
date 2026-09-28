import copy

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

CAMERAS = ('agentview_image', 'in_hand_image')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SpatialSoftmax(nn.Module):
    # Finn et al., Deep Spatial Autoencoders for Visuomotor Learning (robomimic implementation)
    def __init__(self, in_c, in_h, in_w, num_kp=32):
        super().__init__()
        self.in_h, self.in_w, self.num_kp = in_h, in_w, num_kp
        self.nets = nn.Conv2d(in_c, num_kp, kernel_size=1)
        self.register_buffer('temperature', torch.ones(1))
        pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., in_w), np.linspace(-1., 1., in_h))
        self.register_buffer('pos_x', torch.from_numpy(pos_x.reshape(1, in_h * in_w)).float())
        self.register_buffer('pos_y', torch.from_numpy(pos_y.reshape(1, in_h * in_w)).float())

    def forward(self, feature):
        feature = self.nets(feature).reshape(-1, self.in_h * self.in_w)
        attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        return torch.cat([expected_x, expected_y], 1).view(-1, self.num_kp, 2)


class ImageEncoder(nn.Module):
    """ResNet-18 trunk with GroupNorm, spatial softmax over 32 keypoints, linear map to 64."""

    def __init__(self, image_size=96, feature_dim=64):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        side = int(np.ceil(image_size / 32.))
        self.pool = SpatialSoftmax(512, side, side)
        self.linear = nn.Linear(2 * self.pool.num_kp, feature_dim)

    def forward(self, x):
        return self.linear(torch.flatten(self.pool(self.backbone(x)), 1))


def _group_norm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            _group_norm(child)
    return module


class VisualEncoder(nn.Module):
    """The visual encoder of D3IL: one ImageEncoder per camera, features concatenated (N, 128)."""

    def __init__(self):
        super().__init__()
        core = ImageEncoder()
        self.encoders = nn.ModuleDict({k: _group_norm(copy.deepcopy(core)) for k in CAMERAS})
        self.feature_dim = 64 * len(CAMERAS)

    def forward(self, agentview, in_hand):
        """Images (B, T, 3, 96, 96) in [0, 1] -> (B, 128), mean over the T frames."""
        B, T = agentview.shape[:2]
        feats = []
        for k, img in zip(CAMERAS, (agentview, in_hand)):
            img = img.reshape(B * T, *img.shape[2:])
            mean = torch.as_tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            std = torch.as_tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            feats.append(self.encoders[k]((img - mean) / std))
        return torch.cat(feats, dim=-1).view(B, T, -1).mean(dim=1)
