"""
空间感知卷积块（SACB）—— 基于 FFT 的实现。

SACB-Net 的核心贡献。通过 3D FFT 按频率特性（低频主导 vs. 高频主导）对
空间位置进行分区，然后对每组应用不同的动态生成卷积核。替代了原始的 K-Means
聚类方案（见 SACB2.py），实现了确定性、可微分的频率分区。

同时包含 cross_Sim 模块，通过局部互相关注意力估计位移场。
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.nn.modules.utils import _triple
from nn_util import get_act_layer, conv
from einops import rearrange, reduce


def tuple_(x, length=1):
    """将标量 x 重复为指定长度的元组；若已是元组则直接返回。"""
    return x if isinstance(x, tuple) else ((x,) * length)


class FrequencyPartition(nn.Module):
    """通过 FFT 将 3D 特征图分为低频/高频主导区域。

    对每个体素，比较低频分量重建的能量与高频分量的能量。
    高频能量占优的体素标记为 1，否则标记为 0。

    低通截止为归一化频率空间（[-1, 1]^3）中半径为 lp_ratio 的球体。
    掩码按空间形状缓存。
    """

    def __init__(self, lp_ratio=0.15, force_cpu_fft=False):
        super().__init__()
        self.lp_ratio = float(lp_ratio)
        self._mask_cache = {}
        self.force_cpu_fft = bool(force_cpu_fft)

    def set_lp_ratio(self, lp_ratio):
        self.lp_ratio = float(lp_ratio)
        self._mask_cache.clear()

    def set_force_cpu_fft(self, force_cpu_fft):
        self.force_cpu_fft = bool(force_cpu_fft)

    def _build_lowpass_mask(self, shape, device, dtype):
        """在频域中构建球形低通掩码，按形状缓存。"""
        d, h, w = shape
        key = (d, h, w, device.type, device.index, str(dtype), self.lp_ratio)
        if key in self._mask_cache:
            return self._mask_cache[key]

        z = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        dist = torch.sqrt(zz * zz + yy * yy + xx * xx)
        mask = (dist <= self.lp_ratio).to(dtype=dtype)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        self._mask_cache[key] = mask
        return mask

    def _run_partition(self, x):
        """FFT -> 低通掩码分离 -> 比较能量 -> 二值标签。"""
        b, _, d, h, w = x.shape
        mask = self._build_lowpass_mask((d, h, w), x.device, x.dtype)

        x_fft = fft.fftn(x, dim=(-3, -2, -1))
        x_fft = fft.fftshift(x_fft, dim=(-3, -2, -1))

        low_freq = x_fft * mask
        high_freq = x_fft * (1.0 - mask)

        low_spatial = fft.ifftn(fft.ifftshift(low_freq, dim=(-3, -2, -1)), dim=(-3, -2, -1)).abs()
        high_spatial = fft.ifftn(fft.ifftshift(high_freq, dim=(-3, -2, -1)), dim=(-3, -2, -1)).abs()

        low_energy = low_spatial.mean(dim=1)   # [B, D, H, W]
        high_energy = high_spatial.mean(dim=1)  # [B, D, H, W]
        # 0 = low-freq dominant, 1 = high-freq dominant
        return (high_energy > low_energy).long().view(b, -1)

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入特征图 [B, C, D, H, W]。
        Returns:
            cluster_idx: 分区标签 [B, D*H*W]，0=低频主导，1=高频主导。
        """
        x_work = x.contiguous()
        # cuFFT in older CUDA/PyTorch stacks is fragile for reduced precision.
        if x_work.dtype in (torch.float16, torch.bfloat16):
            x_work = x_work.float()

        if not x_work.is_cuda:
            return self._run_partition(x_work)

        if self.force_cpu_fft:
            return self._run_partition(x_work.cpu()).to(x.device)

        try:
            return self._run_partition(x_work)
        except RuntimeError as e:
            if "cuFFT" not in str(e):
                raise
            warnings.warn(
                "cuFFT failed in FrequencyPartition; falling back to CPU FFT for subsequent calls.",
                RuntimeWarning,
            )
            self.force_cpu_fft = True
            return self._run_partition(x_work.cpu()).to(x.device)


class cross_Sim(nn.Module):
    """局部互相关注意力模块，用于位移估计。

    对固定特征（Fy）的每个体素，在运动特征（Fx）的局部窗口内计算
    注意力权重，输出是邻域偏移向量的加权和，得到亚体素级位移估计
    [B, 3, D, H, W]。

    Args:
        win_s: 立方局部窗口的边长（默认 3，即 3x3x3 = 27 个邻居）。
    """

    def __init__(self, win_s=3):
        super(cross_Sim, self).__init__()
        self.wins = win_s
        self.win_len = win_s ** 3

    def forward(self, Fx, Fy, wins=None):
        if wins:
            self.wins = wins
            self.win_len = wins ** 3
        b, c, d, h, w = Fy.shape

        # 构建局部窗口内的相对偏移向量 [win^3, 3]
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [self.wins] * 3]
        grid = torch.stack(torch.meshgrid(vectors, indexing='ij'), -1).type(torch.FloatTensor)
        G = grid.reshape(self.win_len, 3).unsqueeze(0).unsqueeze(0).to(Fx.device)

        # 将固定特征展平为查询向量 [B, N, 1, C]
        Fy = rearrange(Fy, 'b c d h w -> b (d h w) 1 c')
        pd = self.wins // 2

        # 从运动特征中提取局部补丁 [B, N, win^3, C]
        Fx = F.pad(Fx, tuple_(pd, length=6))
        Fx = Fx.unfold(2, self.wins, 1).unfold(3, self.wins, 1).unfold(4, self.wins, 1)
        Fx = rearrange(Fx, 'b c d h w wd wh ww -> b (d h w) (wd wh ww) c')

        # 注意力：query × keys -> softmax -> 加权求和偏移向量
        attn = Fy @ Fx.transpose(-2, -1)
        sim = attn.softmax(dim=-1)
        out = sim @ G
        out = rearrange(out, 'b (d h w) 1 c -> b c d h w', d=d, h=h, w=w)

        return out


class SACB(nn.Module):
    """空间感知卷积块（FFT 频率分区版本）。

    核心流程：
      1. 通过 FFT 将体素分为低频/高频主导两组
      2. 对每组计算中心特征向量（centroid）
      3. 中心特征通过 MLP 生成动态卷积核权重和偏置
      4. 用生成的权重调制基础卷积核，对展开的局部补丁做卷积
      5. 用空间掩码将两组结果合并

    Args:
        in_ch:      输入通道数。
        out_ch:     输出通道数。
        ks:         卷积核大小（3D 中为 ks^3）。
        in_proj_n:  输入投影倍率（1 = 保持通道数）。
        num_k:      分区数（FFT 分区必须为 2）。
        act:        激活函数名称或元组。
        residual:   是否添加残差连接。
        mean_type:  中心特征聚合模式：
                    's' = 对空间/核维度求均值（通道特征），
                    'c' = 对通道求均值（核特征），
                    其他 = 两者拼接。
        lp_ratio:   FrequencyPartition 的低通截止比率。
        n_mlp:      核/偏置生成 MLP 的宽度倍率。
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        ks,
        stride=1,
        in_proj_n=1,
        padding=1,
        dilation=1,
        groups=1,
        num_k=2,
        act='prelu',
        residual=True,
        mean_type='s',
        scale_f=1,
        n_mlp=1,
        sample_n=5,
        m_iter=1e10,
        tol=1e-10,
        fix_rng=False,
        lp_ratio=0.15,
    ):
        super(SACB, self).__init__()
        self.ks = ks
        self.stride = stride
        self.padding = tuple(x for x in reversed(_triple(padding)) for _ in range(2))
        self.dilation = _triple(dilation)
        if int(num_k) != 2:
            raise ValueError(f'FFT SACB only supports num_k=2, got {num_k}')
        self.num_k = 2
        self.res = residual
        self.out_ch = out_ch
        in_ch_n = int(in_ch * in_proj_n)

        # 可学习的基础卷积权重 [out_ch, in_ch_n/groups, ks^3]
        self.w = nn.Parameter(torch.Tensor(out_ch, in_ch_n // groups, self.ks ** 3))
        self.act = get_act_layer(act) if act else None
        self.reset_parameters()
        self.scale_f = scale_f
        self.mean_type = mean_type

        self.freq_partition = FrequencyPartition(lp_ratio=lp_ratio)

        inner_dims = 128 * n_mlp
        inner_dims2 = 64 * n_mlp

        self.sample_n = sample_n
        if mean_type == 's':
            _in_c = in_ch_n
        elif mean_type == 'c':
            _in_c = self.ks ** 3
        else:
            _in_c = in_ch + self.ks ** 3

        # MLP：中心特征 -> 逐元素卷积核调制权重
        self.get_kernel = nn.Sequential(
            nn.Linear(_in_c, inner_dims),
            nn.ReLU(),
            nn.Linear(inner_dims, inner_dims),
            nn.ReLU(),
            nn.Linear(inner_dims, self.ks ** 3),
            nn.Sigmoid(),
        )

        # MLP：中心特征 -> 逐通道偏置
        self.get_bias = nn.Sequential(
            nn.Linear(in_features=_in_c, out_features=inner_dims2),
            nn.ReLU(),
            nn.Linear(in_features=inner_dims2, out_features=inner_dims2),
            nn.ReLU(),
            nn.Linear(in_features=inner_dims2, out_features=out_ch),
        )

        self.proj_in = conv(in_ch, in_ch_n, 3, 1, 1, act=act, norm='instance')

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def set_num_k(self, k):
        """设置分区数（FFT 二值分区必须为 2）。"""
        k_val = int(k)
        if k_val != 2:
            raise ValueError(f'FFT SACB only supports num_k=2, got {k}')
        self.num_k = 2

    def set_lp_ratio(self, lp_ratio):
        """更新低通截止比率并清除掩码缓存。"""
        self.freq_partition.set_lp_ratio(lp_ratio)

    def set_force_cpu_fft(self, force_cpu_fft):
        self.freq_partition.set_force_cpu_fft(force_cpu_fft)

    def scale(self, x, factor, mode='nearest'):
        if mode == 'nearest':
            return F.interpolate(x, scale_factor=factor, mode=mode)
        return F.interpolate(x, scale_factor=factor, mode='trilinear', align_corners=True)

    def feat_mean(self, x, mean_type='s'):
        """聚合展开后补丁特征为每体素的摘要向量。

        Args:
            x: 展开后的补丁 [B, C, nD, nH, nW, k1, k2, k3]。
            mean_type: 's' -> 对核维度求均值（返回 [B, N, C]），
                       'c' -> 对通道求均值（返回 [B, N, k^3]），
                       其他 -> 两者拼接。
        """
        if mean_type == 's':
            x = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) c', 'mean')
        elif mean_type == 'c':
            x = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) (k1 k2 k3)', 'mean')
        else:
            xs = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) c', 'mean')
            xc = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) (k1 k2 k3)', 'mean')
            x = torch.cat([xs, xc], -1)
        return x

    def forward(self, x):
        """应用频率感知动态卷积。

        Args:
            x: 输入特征图 [B, C, D, H, W]。
        Returns:
            输出特征图 [B, out_ch, D, H, W]（空间尺寸不变）。
        """
        b, c, d, h, w = x.shape
        x_in = x                  # 保存输入用于残差连接
        x = self.proj_in(x)       # 输入投影（3x3 卷积 + InstanceNorm + 激活）

        # 展开为重叠的局部补丁 [B, C', D, H, W, ks, ks, ks]
        x_pad = F.pad(x, self.padding)
        x_unfold = x_pad.unfold(2, self.ks, self.stride).unfold(3, self.ks, self.stride).unfold(4, self.ks, self.stride)

        # 计算每体素的特征摘要和 FFT 频率分区标签
        x_mean = self.feat_mean(x_unfold, self.mean_type)
        cluster_idx = self.freq_partition(x)

        # 计算低频/高频两组的中心特征向量
        low_mask = cluster_idx.eq(0).float()      # 低频主导掩码
        high_mask = cluster_idx.eq(1).float()      # 高频主导掩码
        low_denom = low_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        high_denom = high_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        low_centroid = (x_mean * low_mask.unsqueeze(-1)).sum(dim=1) / low_denom
        high_centroid = (x_mean * high_mask.unsqueeze(-1)).sum(dim=1) / high_denom
        centroids = [low_centroid, high_centroid]

        # 重排补丁用于批量矩阵乘法 [B, C'*ks^3, N]
        x = rearrange(x_unfold, 'b c nd nh nw k1 k2 k3 -> b (c k1 k2 k3) (nd nh nw)')
        out = x.new_zeros(b, self.out_ch, d * h * w)

        # 对每个频率组：生成动态核/偏置，卷积，掩码选择性叠加
        for i in range(self.num_k):
            mask = cluster_idx.eq(i).float()
            if mask.sum() == 0:
                continue
            cat_ = centroids[i]
            weight = rearrange(self.get_kernel(cat_), 'b k -> b 1 1 k') * self.w.unsqueeze(0)
            bias = rearrange(self.get_bias(cat_), 'b o -> b o 1')
            response = torch.einsum('b i j, b o i -> b o j', x, rearrange(weight, 'b o i k -> b o (i k)')) + bias
            out = out + response * mask.unsqueeze(1)

        out = rearrange(out, 'b o (d h w) -> b o d h w', d=d, h=h, w=w)
        if self.act:
            out = self.act(out)
        if self.res:
            out = out + x_in

        return out
