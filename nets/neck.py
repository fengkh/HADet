import torch
import torch.nn as nn
from basic_modules import CBS
from fca_block import FCABlock
from csa_block import CSABlock


class PAFPN(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], n=3):
        super(PAFPN, self).__init__()
        self.lateral_conv0 = CBS(
            in_channels[2], in_channels[1], kernel_size=1, stride=1, padding=0
        )
        self.top_down_block1 = FCABlock(in_channels[1] * 2, in_channels[1], n=n)

        self.reduce_conv1 = CBS(
            in_channels[1], in_channels[0], kernel_size=1, stride=1, padding=0
        )
        self.top_down_block2 = FCABlock(in_channels[0] * 2, in_channels[0], n=n)

        self.bu_conv2 = CBS(
            in_channels[0], in_channels[0], kernel_size=3, stride=2, padding=1
        )
        self.bottom_up_block1 = CSABlock(in_channels[0] * 2, in_channels[1], n=n, r=4)

        self.bu_conv1 = CBS(
            in_channels[1], in_channels[1], kernel_size=3, stride=2, padding=1
        )
        self.bottom_up_block2 = CSABlock(in_channels[1] * 2, in_channels[2], n=n, r=2)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, features):
        f3, f4, f5 = features

        lat_f5 = self.lateral_conv0(f5)
        up_f5 = self.upsample(lat_f5)
        concat_f4 = torch.cat([up_f5, f4], 1)
        p4 = self.top_down_block1(concat_f4)

        lat_p4 = self.reduce_conv1(p4)
        up_p4 = self.upsample(lat_p4)
        concat_f3 = torch.cat([up_p4, f3], 1)
        p3 = self.top_down_block2(concat_f3)

        down_p3 = self.bu_conv2(p3)
        concat_n4 = torch.cat([down_p3, lat_p4], 1)
        n4 = self.bottom_up_block1(concat_n4)

        down_n4 = self.bu_conv1(n4)
        concat_n5 = torch.cat([down_n4, lat_f5], 1)
        n5 = self.bottom_up_block2(concat_n5)

        return p3, n4, n5
