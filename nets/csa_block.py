import torch
import torch.nn as nn
from nets.basic_modules import CBS, GSBN


class CSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, n=3, r=2):
        super(CSABlock, self).__init__()
        self.split_c = out_channels // 2
        self.r = r

        self.cbs_init = CBS(in_channels, out_channels, kernel_size=1)
        self.gsbn_blocks = nn.Sequential(*[GSBN(self.split_c) for _ in range(n)])

        self.cbs_att1 = CBS(self.split_c, self.split_c, kernel_size=3)
        self.cbs_att2 = CBS(self.split_c, self.split_c, kernel_size=3)

        self.channel_mean_pool = nn.Conv2d(self.split_c, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.channel_mean_pool.weight, 1.0 / self.split_c)
        for param in self.channel_mean_pool.parameters():
            param.requires_grad = False

        self.softmax = nn.Softmax(dim=-1)
        self.cbs_out = CBS(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, W, H = x.shape

        x = self.cbs_init(x)
        x_hat_csa, x_bar_csa = torch.split(x, self.split_c, dim=1)

        x_csa = self.gsbn_blocks(x_hat_csa) + x_hat_csa

        x_bar_in = self.cbs_att1(x_bar_csa)
        x_bar_in = self.cbs_att2(x_bar_in + x_bar_csa)

        s = self.channel_mean_pool(x_bar_in)

        cell_w, cell_h = W // self.r, H // self.r
        s_partitioned = s.view(B, 1, self.r, cell_w, self.r, cell_h)
        s_partitioned = s_partitioned.permute(0, 1, 2, 4, 3, 5).contiguous()
        s_flatten = s_partitioned.view(B, 1, self.r, self.r, -1)

        s_attn = self.softmax(s_flatten)

        s_attn = s_attn.view(B, 1, self.r, self.r, cell_w, cell_h)
        s_attn = s_attn.permute(0, 1, 2, 4, 3, 5).contiguous()
        s_attn = s_attn.view(B, 1, W, H)

        x_bar_out = x_bar_in * s_attn

        out = torch.cat([x_csa, x_bar_out], dim=1)
        out = self.cbs_out(out)

        return out
