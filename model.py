"""
SACB-Net: 空间感知卷积用于医学图像配准。

定义配准网络主体结构：共享的 5 级特征金字塔编码器 + 从粗到精的解码器，
解码器使用 SACB 模块（频率感知动态卷积）和交叉相似度光流估计。

参考文献: Cheng et al., CVPR 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from nn_util import get_act_layer, conv, unfoldNd, tuple_
from SACB1 import SACB, cross_Sim
import utils


class Encoder(nn.Module):
    """五级特征金字塔编码器。

    生成五个分辨率级别的特征图（1x, 1/2, 1/4, 1/8, 1/16），
    通道数递增：c -> 2c -> 4c -> 8c -> 16c -> 16c。
    每级由双卷积块组成，除第一级外均有 AvgPool3d 下采样。

    Args:
        in_c: 输入通道数（单模态灰度图为 1）。
        c:    基础通道数，后续各级为 c 的倍数。
    """

    def __init__(self, in_c=1, c=4):
        super(Encoder, self).__init__()

        act=("leakyrelu", {"negative_slope": 0.1})  # 激活函数：LeakyReLU
        norm= 'instance'  # 归一化方式：InstanceNorm
        # 第1级：输入 -> 2c 通道（全分辨率）
        self.conv0 = double_conv(c, 2*c, act=act, norm=norm,
                        pre_fn=conv(in_c,c,act=act,norm=norm))
        # 第2级：2c -> 4c（1/2 分辨率）
        self.conv1 =  double_conv(2 * c, 4 * c, act=act, norm=norm,
                        pre_fn=nn.AvgPool3d(2))
        # 第3级：4c -> 8c（1/4 分辨率）
        self.conv2 = double_conv(4 * c, 8 * c, act=act, norm=norm,
                        pre_fn= nn.AvgPool3d(2))
        # 第4级：8c -> 16c（1/8 分辨率）
        self.conv3 = double_conv(8 * c, 16* c, act=act, norm=norm,
                        pre_fn=nn.AvgPool3d(2))
        # 第5级：16c -> 16c（1/16 分辨率）
        self.conv4 =double_conv(16 * c, 16 * c, act=act, norm=norm,
                        pre_fn=nn.AvgPool3d(2))

    def forward(self, x):
        out0 = self.conv0(x)  # 1x   resolution, 2c channels
        out1 = self.conv1(out0)  # 1/2  resolution, 4c channels
        out2 = self.conv2(out1)  # 1/4  resolution, 8c channels
        out3 = self.conv3(out2)  # 1/8  resolution, 16c channels
        out4 = self.conv4(out3)  # 1/16 resolution, 16c channels

        return out0, out1, out2, out3, out4


def double_conv(in_c, out_c, act, norm='instance', append_fn=None, pre_fn=None):
    """两个连续的 3x3x3 卷积层，可选前置/后置操作。

    作为编码器和解码器各级的基本构建块。
    """
    layer = nn.Sequential(pre_fn if pre_fn else nn.Identity(),
                          conv(in_c,  out_c, 3,1,1,act=act,norm=norm),
                          conv(out_c, out_c, 3,1,1,act=act,norm=norm),
                          append_fn if append_fn else nn.Identity()
                        )
    return layer


class SACB_Net(nn.Module):
    """SACB-Net 配准网络。

    网络架构概览：
      1. 共享编码器从运动图像和固定图像中提取五级分辨率的特征。
      2. 从最粗级（1/16）到最精级（1x），SACB 模块利用频率感知动态卷积
         精炼特征，交叉相似度模块估计位移增量。
      3. 位移场在各尺度间组合并上采样，最终输出全分辨率的密集位移场（Phi）。

    Args:
        inshape:    输入体数据的空间尺寸 (D, H, W)。
        in_c:       输入通道数（灰度图为 1）。
        ch_scale:   基础通道倍率 c；编码器通道为该值的倍数。
        num_k:      每个 SACB 的频率分区数（FFT 分区必须为 2）。
                    标量或 4 元组用于逐尺度控制。
        lp_ratio:   FFT 频率分区的低通半径比率。
                    标量或 4 元组用于逐尺度控制。
        scale:      保留的缩放因子（当前未使用）。
        mean_type:  SACB 中的特征聚合策略：
                    's' = 空间/通道均值, 'c' = 核均值, 其他 = 两者兼用。
    """

    def __init__(self,
                 inshape=(160,192,160),
                 in_c = 1,
                 ch_scale = 4,
                 num_k = 2,
                 lp_ratio = 0.15,
                 scale = 1.,
                 mean_type='s'
                ):
        super(SACB_Net, self).__init__()
        self.ch_scale = ch_scale
        self.inshape = inshape
        self.scale = scale
        c = self.ch_scale
        self.mt = mean_type
        if type(num_k) is not tuple:
            self.num_k = tuple_(num_k, length=4)
        else:
            self.num_k = num_k
        if type(lp_ratio) is not tuple:
            self.lp_ratio = tuple_(lp_ratio, length=4)
        else:
            self.lp_ratio = lp_ratio
        self.encoder = Encoder(in_c=in_c, c=c)
        act=("leakyrelu", {"negative_slope": 0.1})

        proj_n = 1  # 输入投影倍率
        self.up_tri = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)  # 2倍三线性上采样

        self.conv1 = double_conv(2*c, 2*c, act=act)         # 最精级的普通双卷积（不使用 SACB）
        self.cross_sim = cross_Sim()                         # 交叉相似度位移估计模块

        # 各尺度的 SACB 模块（频率感知动态卷积）
        self.sacb_proj2 = SACB(4*c,   4*c, in_proj_n=proj_n, ks=3, mean_type=self.mt, num_k=self.num_k[0], act=act, residual=True, lp_ratio=self.lp_ratio[0])
        self.sacb_proj3 = SACB(8*c,   8*c, in_proj_n=proj_n, ks=3, mean_type=self.mt, num_k=self.num_k[1], act=act, residual=True, lp_ratio=self.lp_ratio[1])
        self.sacb_proj4 = SACB(16*c, 16*c, in_proj_n=proj_n, ks=3, mean_type=self.mt, num_k=self.num_k[2], act=act, residual=True, lp_ratio=self.lp_ratio[2])
        self.sacb_proj5 = SACB(16*c, 16*c, in_proj_n=proj_n, ks=3, mean_type=self.mt, num_k=self.num_k[3], act=act, residual=True, lp_ratio=self.lp_ratio[3])

        # 最精级输出头：拼接固定/运动特征 -> 双卷积 -> 3 通道位移场
        self.conv1_out = double_conv(2*2*c, 2*c, act=act, append_fn=conv(2*c,3, 3,1,1, act=None))

        # 各尺度的空间变换器（用于位移场组合和图像变形）
        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(utils.SpatialTransformer([s // 2**i for s in inshape]))

    def set_k(self, k):
        """设置所有 SACB 模块的频率分区数（必须为 2）。"""
        if type(k) is not tuple:
            k = tuple_(k, length=4)
        if any(int(v) != 2 for v in k):
            raise ValueError(f'FFT SACB only supports num_k=2 per scale, got {k}')
        self.sacb_proj2.set_num_k(k[0])
        self.sacb_proj3.set_num_k(k[1])
        self.sacb_proj4.set_num_k(k[2])
        self.sacb_proj5.set_num_k(k[3])

    def set_lp_ratio(self, lp_ratio):
        """设置所有 SACB 模块的 FFT 低通截止比率。"""
        if type(lp_ratio) is not tuple:
            lp_ratio = tuple_(lp_ratio, length=4)
        self.sacb_proj2.set_lp_ratio(lp_ratio[0])
        self.sacb_proj3.set_lp_ratio(lp_ratio[1])
        self.sacb_proj4.set_lp_ratio(lp_ratio[2])
        self.sacb_proj5.set_lp_ratio(lp_ratio[3])

    def set_force_cpu_fft(self, force_cpu_fft):
        """强制所有 SACB 模块在 CPU 上运行 FFT（用于解决 cuFFT 兼容性问题）。"""
        self.sacb_proj2.set_force_cpu_fft(force_cpu_fft)
        self.sacb_proj3.set_force_cpu_fft(force_cpu_fft)
        self.sacb_proj4.set_force_cpu_fft(force_cpu_fft)
        self.sacb_proj5.set_force_cpu_fft(force_cpu_fft)

    def forward(self, x, y, softsign_last=False):
        """执行多尺度配准：从运动图像 x 到固定图像 y。

        Args:
            x: 运动图像 [B, 1, D, H, W]。
            y: 固定图像 [B, 1, D, H, W]。
            softsign_last: 若为 True，对最精级残差流施加 softsign 以限制幅值。

        Returns:
            x_warped: 用预测位移场变形后的运动图像 [B, 1, D, H, W]。
            Phi:      全分辨率位移场 [B, 3, D, H, W]。
        """
        # --- 编码阶段：共享编码器分别提取两幅图像的五级特征 ---
        M1, M2, M3, M4, M5 = self.encoder(x)  # 运动图像特征
        F1, F2, F3, F4, F5 = self.encoder(y)  # 固定图像特征

        # --- 最粗级（1/16）：SACB 特征精炼 + 初始光流估计 ---
        F5, M5 = self.sacb_proj5(F5), self.sacb_proj5(M5)  # SACB 特征精炼
        phi_5 = self.cross_sim(M5, F5)                       # 交叉相似度估计初始位移
        phi_5 = self.up_tri(2.* phi_5)                        # 上采样到 1/8 分辨率（位移值×2）

        # --- 1/8 尺度：用粗级位移变形运动特征，SACB 精炼，组合位移 ---
        M4 = self.transformer[3](M4, phi_5)                   # 用 phi_5 变形 M4
        F4, M4 = self.sacb_proj4(F4), self.sacb_proj4(M4)
        delta_phi_4 = self.cross_sim(M4, F4)                  # 估计残差位移
        phi_4 = self.up_tri(2.* (self.transformer[3](phi_5, delta_phi_4) + delta_phi_4))  # 组合并上采样

        # --- 1/4 尺度 ---
        M3 = self.transformer[2](M3, phi_4)
        F3, M3 = self.sacb_proj3(F3), self.sacb_proj3(M3)
        delta_phi_3 = self.cross_sim(M3, F3)
        phi_3 = self.up_tri(2.* (self.transformer[2](phi_4, delta_phi_3) + delta_phi_3))

        # --- 1/2 尺度 ---
        M2 = self.transformer[1](M2, phi_3)
        F2, M2 = self.sacb_proj2(F2), self.sacb_proj2(M2)
        delta_phi_2 = self.cross_sim(M2, F2)
        phi_2 = self.up_tri(2.* (self.transformer[1](phi_3, delta_phi_2) + delta_phi_2))

        # --- 最精级（1x）：普通卷积（不使用 SACB），最终残差流 ---
        M1 = self.transformer[0](M1, phi_2)
        F1, M1 = self.conv1(F1), self.conv1(M1)
        delta_phi_1 = self.conv1_out(torch.cat([F1, M1],1))  # 拼接后输出 3 通道位移场
        if softsign_last:
            delta_phi_1 = F.softsign(delta_phi_1)

        # 将上采样的粗级位移与最精级残差组合，得到最终位移场
        Phi = self.transformer[0](phi_2, delta_phi_1) + delta_phi_1

        x_warped = self.transformer[0](x, Phi)  # 用最终位移场变形运动图像
        return x_warped, Phi
