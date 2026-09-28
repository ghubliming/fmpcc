import math

import einops
import torch
from einops.layers.torch import Rearrange
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


def _embedding(dim):
    return nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))


class TemporalUnet(nn.Module):
    """Temporal U-Net of Diffuser/DPCC over plans (B, H, d).

    two_time: interval embedding h added to the time embedding, and a second output head (the
    instantaneous velocity of the average-velocity models). cond_dim > 0: a conditioning vector
    (the visual feature) is projected and concatenated with the time embedding.
    """

    def __init__(self, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), two_time=False, cond_dim=0, kernel_size=5):
        super().__init__()
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = _embedding(dim)
        if two_time:
            self.h_mlp = _embedding(dim)
        self.cond_mlp = None
        embed_dim = dim
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, dim), nn.Mish(), nn.Linear(dim, dim))
            embed_dim = 2 * dim
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim, kernel_size),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim, kernel_size),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, kernel_size)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, kernel_size)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim, kernel_size),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim, kernel_size),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))
        self.final_conv = nn.Sequential(Conv1dBlock(dim, dim, kernel_size), nn.Conv1d(dim, transition_dim, 1))
        if two_time:
            self.v_final_conv = nn.Sequential(Conv1dBlock(dim, dim, kernel_size), nn.Conv1d(dim, transition_dim, 1))

    def forward(self, x, time, h=None, cond=None, return_v=False):
        x = einops.rearrange(x, 'b h t -> b t h')
        t = self.time_mlp(time)
        if h is not None:
            t = t + self.h_mlp(h)
        if self.cond_mlp is not None:
            t = torch.cat([t, self.cond_mlp(cond)], dim=-1)
        skips = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            skips.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)
        u = einops.rearrange(self.final_conv(x), 'b t h -> b h t')
        if return_v:
            return u, einops.rearrange(self.v_final_conv(x), 'b t h -> b h t')
        return u
