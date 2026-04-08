import torch
import torch.nn as nn
from backbone import BA3DetBackbone
from Neck import PAFPN
from basic_modules import CBS


class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(DecoupledHead, self).__init__()
        self.stem = CBS(in_channels, in_channels, kernel_size=1)

        self.cls_convs = nn.Sequential(
            CBS(in_channels, in_channels, kernel_size=3, padding=1),
            CBS(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.reg_convs = nn.Sequential(
            CBS(in_channels, in_channels, kernel_size=3, padding=1),
            CBS(in_channels, in_channels, kernel_size=3, padding=1),
        )

        self.cls_preds = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.reg_preds = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.obj_preds = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)

        cls_out = self.cls_preds(cls_feat)
        reg_out = self.reg_preds(reg_feat)
        obj_out = self.obj_preds(reg_feat)

        return cls_out, reg_out, obj_out


class BA3Det(nn.Module):
    def __init__(self, num_classes=1):
        super(BA3Det, self).__init__()
        self.backbone = BA3DetBackbone()
        self.neck = PAFPN(in_channels=[256, 512, 1024])

        self.head_1 = DecoupledHead(256, num_classes)
        self.head_2 = DecoupledHead(512, num_classes)
        self.head_3 = DecoupledHead(1024, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        fpn_outs = self.neck(features)

        outs = []
        outs.append(self.head_1(fpn_outs[0]))
        outs.append(self.head_2(fpn_outs[1]))
        outs.append(self.head_3(fpn_outs[2]))

        return outs
