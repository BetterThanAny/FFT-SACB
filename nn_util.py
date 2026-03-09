"""
3D 医学图像配准的神经网络工具层。

提供以下功能：
  - unfoldNd: N 维展开操作
  - conv: MONAI 卷积封装
  - tuple_: 标量→元组辅助函数
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair, _single, _triple
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act
from monai.networks.layers.utils import get_act_layer

Act.add_factory_callable("softsign", lambda: nn.modules.Softsign)

def exists(x): return x is not None


def tuple_(x, length=1):
    """将标量 x 重复为指定长度的元组；若已是元组则直接返回。"""
    return x if isinstance(x, tuple) else ((x,) * length)


def _make_weight(in_channels, kernel_size, device, dtype):
    """构建单位卷积核（one-hot 恒等核），用于深度可分离展开。

    创建 [C * prod(kernel_size), 1, *kernel_size] 形状的权重张量，
    每个输出通道精确提取局部补丁中的一个元素。
    """
    kernel_size_numel =  int(np.prod((kernel_size)))
    repeat = [in_channels, 1] + [1 for _ in kernel_size]

    return (
        torch.eye(kernel_size_numel, device=device, dtype=dtype)
        .reshape((kernel_size_numel, 1, *kernel_size))
        .repeat(*repeat)
    )


def unfoldNd(input, kernel_size, dilation=1, padding=0, stride=1):
    """N 维展开操作，使用深度卷积和单位核实现。

    等价于 torch.Tensor.unfold，但支持任意空间维度（1D/2D/3D）的统一接口。

    Args:
        input: 输入张量 [B, C, *spatial]。
        kernel_size: 滑动窗口大小。
        dilation, padding, stride: 标准卷积参数。
    Returns:
        展开后的张量 [B, C * prod(kernel_size), *output_spatial]。
    """
    b,c  = input.shape[0], input.shape[1]
    # get convolution operation
    spatial_dim = input.dim() - 2
    conv = [F.conv1d, F.conv2d, F.conv3d][spatial_dim-1]
    _tuple = [_pair, _single, _triple]

    kernel_size =_tuple[spatial_dim-1](kernel_size)
    # prepare one-hot convolution kernel
    weight = _make_weight(c, kernel_size, input.device, input.dtype)

    unfold = conv(
        input,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=c,
    )

    return unfold


def conv(in_c, out_c, k=3, s=1, p=1, dim=3, bias=True,
         act='prelu', norm=None, drop=None):
    """MONAI Convolution 的便捷封装。

    为 3D 配准网络提供合理默认值：kernel_size=3, stride=1, padding=1, PReLU 激活。
    """
    return Convolution(
            spatial_dims=dim,
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            strides=s,
            padding=p,
            adn_ordering="NA" if exists(norm) else "A",
            act=act,
            dropout=drop,
            norm=norm,
            bias=bias,
        )
