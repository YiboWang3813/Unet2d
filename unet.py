from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DoubleConv(nn.Sequential):
    """ 定义双卷积层 """
    def __init__(self, in_channels, out_channels, mid_channels=None): 
        # type: (int, int, int) -> None 
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        )


class Down(nn.Sequential): 
    """ 定义下采样模块 最大池化+双卷积 """
    def __init__(self, in_channels, out_channels): 
        # type: (int, int) -> None 
        super().__init__(
            nn.MaxPool2d(kernel_size=2, stride=2), 
            DoubleConv(in_channels, out_channels) 
        )


class Up(nn.Module): 
    """ 定义上采样模块 上采样+拼接+双卷积 """
    def __init__(self, in_channels, out_channels, bilinear=True): 
        # type: (int, int, bool) -> None 
        super(Up, self).__init__()

        if bilinear: 
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) 
        else: 
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) 
            self.conv = DoubleConv(in_channels, out_channels)  

    def forward(self, x1, x2): 
        # type: (Tensor, Tensor) -> Tensor 
        x1 = self.up(x1) 

        # 防止输入的图像不是16的整数倍,进行pad [N, C, H, W]
        diff_h = x2.size()[2] - x1.size()[2] 
        diff_w = x2.size()[3] - x1.size()[3] 

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, 
                        diff_h // 2, diff_h - diff_h // 2]) 

        x = torch.cat([x1, x2], dim=1) 
        x = self.conv(x) 

        return x  


class OutConv(nn.Sequential): 
    def __init__(self, in_channels, num_classes): 
        # type: (int, int) -> None 
        super().__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1) 
        )


class Unet(nn.Module): 
    def __init__(self, 
                 in_channels: int = 1, 
                 num_classes: int = 2, 
                 bilinear: bool = True, 
                 base_c: int = 32): 
        super(Unet, self).__init__()

        self.in_conv = DoubleConv(in_channels, base_c) 
        self.down1 = Down(base_c, base_c * 2) 
        self.down2 = Down(base_c * 2, base_c * 4) 
        self.down3 = Down(base_c * 4, base_c * 8) 
        # 如果使用双线性插值,Up时通道不减半,就要提前限制通道大小为原来的一半 
        # 如果不使用双线性插值,Up时通道自动减半,就可以保留较大的通道输入
        # 但无论哪种,cat都会使得输入通道倍增,同时输出通道要和上一级的通道匹配 
        factor = 2 if bilinear else 1 
        self.down4 = Down(base_c * 8, base_c * 16 // factor) 
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear) 
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear) 
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear) 
        self.up4 = Up(base_c * 2, base_c, bilinear) 
        self.out_conv = OutConv(base_c, num_classes) 

    def forward(self, x): 
        # type: (Tensor) -> Dict[str, Tensor] 
        x1 = self.in_conv(x) 
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4) 
        x = self.up1(x5, x4) 
        x = self.up2(x, x3) 
        x = self.up3(x, x2) 
        x = self.up4(x, x1) 
        logits = self.out_conv(x) 

        return {'out': logits} 
