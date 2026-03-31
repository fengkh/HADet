import torch.nn as nn
from basic_modules import CBS, Focus
from fca_block import FCABlock
from csa_block import CSABlock


class HADetBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(HADetBackbone, self).__init__()
        self.stem = Focus(in_channels, base_channels)

        self.dark2 = nn.Sequential(
            CBS(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            FCABlock(base_channels * 2, base_channels * 2, n=3),
        )

        self.dark3 = nn.Sequential(
            CBS(
                base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1
            ),
            FCABlock(base_channels * 4, base_channels * 4, n=9),
        )

        self.dark4 = nn.Sequential(
            CBS(
                base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1
            ),
            CSABlock(base_channels * 8, base_channels * 8, n=9, r=4),
        )

        self.dark5 = nn.Sequential(
            CBS(
                base_channels * 8,
                base_channels * 16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            CSABlock(base_channels * 16, base_channels * 16, n=3, r=2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        f3 = self.dark3(x)
        f4 = self.dark4(f3)
        f5 = self.dark5(f4)
        return f3, f4, f5
