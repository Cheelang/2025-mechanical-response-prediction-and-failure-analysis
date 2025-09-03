import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNRegressor(nn.Module):
    """
    通用 FCN -> 回归标量
    参数:
        in_chans: 输入通道 (RGB=3)
        base_ch: 第一层通道数
        depth: 卷积下采样层数
    """
    def __init__(self, in_chans=3, base_ch=32, depth=3, num_classes=1):
        super(FCNRegressor, self).__init__()

        layers = []
        in_c = in_chans
        for d in range(depth):
            out_c = base_ch * (2 ** d)
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c

        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(in_c, num_classes)

    def forward(self, x):
        x = self.features(x)  # [B,C,H/2^depth,W/2^depth]
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)
