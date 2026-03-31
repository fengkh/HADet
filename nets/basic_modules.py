import torch
import torch.nn as nn


class CBS(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1
    ):
        super(CBS, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class GSBN(nn.Module):
    def __init__(self, channels, groups=32):
        super(GSBN, self).__init__()
        g = groups if channels % groups == 0 else 1
        self.cbs1 = CBS(channels, channels, kernel_size=3, padding=1, groups=g)
        self.shuffle = ChannelShuffle(groups=g)
        self.cbs2 = CBS(channels, channels, kernel_size=5, padding=2, groups=g)

    def forward(self, x):
        residual = x
        out = self.cbs1(x)
        out = self.shuffle(out)
        out = self.cbs2(out)
        return out + residual


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Focus, self).__init__()
        self.conv = CBS(
            in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1
        )
        return self.conv(x)
