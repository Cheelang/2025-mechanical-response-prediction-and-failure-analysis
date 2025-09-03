import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetRegressor(nn.Module):
    def __init__(self, in_chans=3, num_classes=1, base_ch=64, depth=4):
        """
        通用UNet回归器
        :param in_chans: 输入通道数 (默认 3)
        :param num_classes: 输出通道数 (这里是 1, 表示标量回归)
        :param base_ch: 最初卷积通道数 (默认 64)
        :param depth: U-Net 下采样层数 (默认 4)
        """
        super(UNetRegressor, self).__init__()

        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_c = in_chans
        for d in range(depth):
            out_c = base_ch * (2 ** d)
            self.encoders.append(UNetBlock(in_c, out_c))
            self.pools.append(nn.MaxPool2d(2))
            in_c = out_c

        # bottleneck
        self.bottleneck = UNetBlock(in_c, in_c * 2)
        bottleneck_c = in_c * 2

        # 解码器
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_c = bottleneck_c
        for d in reversed(range(depth)):
            out_c = base_ch * (2 ** d)
            self.ups.append(nn.ConvTranspose2d(prev_c, out_c, kernel_size=2, stride=2))
            self.decoders.append(UNetBlock(out_c * 2, out_c))  # 拼接后通道翻倍
            prev_c = out_c

        # 最后卷积 + 全局池化 + 全连接
        self.final_conv = nn.Conv2d(base_ch, base_ch, kernel_size=1)
        self.fc = nn.Linear(base_ch, num_classes)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.size() != skip.size():  # 尺寸对齐（padding）
                diffY = skip.size()[2] - x.size()[2]
                diffX = skip.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        x = self.final_conv(x)  # [B, base_ch, H, W]
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, base_ch, 1, 1]
        x = x.view(x.size(0), -1)  # [B, base_ch]
        return self.fc(x).squeeze(-1)
