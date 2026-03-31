import torch
import torch.nn as nn
from nets.basic_modules import CBS, GSBN


class FCABlock(nn.Module):
    def __init__(self, in_channels, out_channels, n=3):
        super(FCABlock, self).__init__()
        self.split_c = out_channels // 2

        self.cbs_init = CBS(in_channels, out_channels, kernel_size=1)
        self.gsbn_blocks = nn.Sequential(*[GSBN(self.split_c) for _ in range(n)])

        self.cbs_att1 = CBS(self.split_c, self.split_c, kernel_size=3)
        self.cbs_att2 = CBS(self.split_c, self.split_c, kernel_size=3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Linear(self.split_c * 2, self.split_c)
        self.sigmoid = nn.Sigmoid()

        self.cbs_out = CBS(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.cbs_init(x)
        x_bar_fca, x_hat_fca = torch.split(x, self.split_c, dim=1)

        x_fca = self.gsbn_blocks(x_bar_fca) + x_bar_fca

        x_hat_in = self.cbs_att1(x_hat_fca)
        x_hat_in = self.cbs_att2(x_hat_in + x_hat_fca)

        u_hat = self.avg_pool(x_hat_in).squeeze(-1).squeeze(-1)
        u_bar = self.max_pool(x_hat_in).squeeze(-1).squeeze(-1)

        fft_avg = torch.fft.fft(u_hat, dim=1)
        fft_max = torch.fft.fft(u_bar, dim=1)

        mag_avg = torch.abs(fft_avg)
        mag_max = torch.abs(fft_max)

        concat_mag = torch.cat([mag_avg, mag_max], dim=1)
        att_weights = self.sigmoid(self.fc(concat_mag)).unsqueeze(-1).unsqueeze(-1)
        x_hat_out = x_hat_in * att_weights

        out = torch.cat([x_fca, x_hat_out], dim=1)
        out = self.cbs_out(out)

        return out
