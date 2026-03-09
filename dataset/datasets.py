"""
医学图像配准数据集模块。

为 SACB-Net 训练提供以下数据集实现：
  - IXIBrainDataset / IXIBrainInferDataset: IXI 脑部数据集（图谱到受试者配准）
  - OASISBrainInferDataset: OASIS 脑部数据集（验证用）
  - LPBABrainDatasetS2S / LPBABrainInferDatasetS2S: LPBA40 脑部数据集（受试者间配准）

训练集返回 (moving, fixed) 图像对；
验证集额外返回 (moving_seg, fixed_seg) 分割标签用于 Dice 评估。
"""

import os, glob
import torch, sys
from torch.utils.data import Dataset
import numpy as np
from .data_utils import pkload


def _pair_from_index(paths, num_paths, index):
    """根据线性索引计算受试者间配对的（源路径, 目标路径）。

    将所有 N*(N-1) 个非自身配对展平为一维索引，避免自配对。

    Args:
        paths: 数据文件路径列表。
        num_paths: 路径数量。
        index: 线性配对索引。
    Returns:
        (src_path, tgt_path) 元组。
    """
    n = num_paths
    if n < 2:
        raise IndexError('Need at least 2 samples to form source-target pairs.')
    max_index = n * (n - 1)
    if index < 0 or index >= max_index:
        raise IndexError(f'Pair index out of range: {index}')
    src_idx = index // (n - 1)
    tgt_idx = index % (n - 1)
    if tgt_idx >= src_idx:
        tgt_idx += 1
    return paths[src_idx], paths[tgt_idx]


class IXIBrainDataset(Dataset):
    """IXI 脑部训练数据集（图谱到受试者配准）。

    将固定的图谱图像（atlas）与每个受试者图像配对。
    返回 (atlas_image, subject_image)，均为 [1, D, H, W] 的 float32 张量。

    Args:
        data_path: 受试者 pkl 文件路径列表。
        atlas_path: 图谱 pkl 文件路径。
        transforms: 数据增强/类型转换的组合变换。
    """
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms
        self.atlas_img, self.atlas_seg = pkload(self.atlas_path)

    def one_hot(self, img, C):
        """将标签图转换为 one-hot 编码 [C, H, W, D]。"""
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        """返回第 index 个样本：(atlas_image, subject_image)。"""
        path = self.paths[index]
        x = self.atlas_img.copy()       # 运动图像（图谱）
        y, y_seg = pkload(path)          # 固定图像（受试者）

        x, y = x[None, ...], y[None, ...]  # 添加通道维 [1, D, H, W]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x)  # 确保内存连续，提升数据加载效率
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    """IXI 脑部验证数据集（图谱到受试者配准）。

    与训练集的区别是额外返回分割标签，用于计算 Dice 分数。
    返回 (atlas_image, subject_image, atlas_seg, subject_seg)。

    Args:
        data_path: 受试者 pkl 文件路径列表。
        atlas_path: 图谱 pkl 文件路径。
        transforms: 数据变换。
    """
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms
        self.atlas_img, self.atlas_seg = pkload(self.atlas_path)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x = self.atlas_img.copy()
        x_seg = self.atlas_seg.copy()
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
    

class OASISBrainInferDataset(Dataset):
    """OASIS 脑部验证数据集。

    每个 pkl 文件中包含 (moving, fixed, moving_seg, fixed_seg) 四个数组。
    返回 (moving, fixed, moving_seg, fixed_seg)。
    """
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
    

class LPBABrainDatasetS2S(Dataset):
    """LPBA40 脑部训练数据集（受试者间配准, Subject-to-Subject）。

    将 N 个受试者两两配对（排除自配对），共 N*(N-1) 个训练样本。
    返回 (moving_image, fixed_image)。

    Args:
        data_path: 受试者 pkl 文件路径列表。
        transforms: 数据变换。
    """
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms
        self.num_paths = len(self.paths)

    def _pair_from_index(self, index):
        return _pair_from_index(self.paths, self.num_paths, index)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        """返回第 index 个配对样本：(moving_image, fixed_image)。"""
        path_x, path_y = self._pair_from_index(index)
        x, x_seg = pkload(path_x)  # 运动图像
        y, y_seg = pkload(path_y)   # 固定图像

        x, y = x[None, ...], y[None, ...]  # 添加通道维
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x)  # 确保内存连续
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        """数据集大小 = N * (N-1)，即所有非自身配对数。"""
        return self.num_paths * (self.num_paths - 1)


class LPBABrainInferDatasetS2S(Dataset):
    """LPBA40 脑部验证数据集（受试者间配准）。

    与训练集的区别是额外返回分割标签用于 Dice 评估。
    返回 (moving, fixed, moving_seg, fixed_seg)。
    """
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms
        self.num_paths = len(self.paths)

    def _pair_from_index(self, index):
        return _pair_from_index(self.paths, self.num_paths, index)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        """返回第 index 个配对样本及其分割标签。"""
        path_x, path_y = self._pair_from_index(index)
        x, x_seg = pkload(path_x)  # 运动图像及其分割
        y, y_seg = pkload(path_y)   # 固定图像及其分割
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return self.num_paths * (self.num_paths - 1)
