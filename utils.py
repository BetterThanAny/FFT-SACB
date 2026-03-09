"""
医学图像配准通用工具模块。

包含以下功能：
  - AverageMeter: 训练/验证指标的均值和标准差统计
  - SpatialTransformer: N 维空间变换器（基于位移场的图像变形）
  - register_model: 将空间变换器封装为独立模块
  - dice_val / dice_val_VOI / dice_LPBA / dice_abdo: 各数据集的 Dice 分数计算
  - jacobian_determinant_vxm: 位移场的雅可比行列式计算（评估形变质量）
  - process_label / dice_val_substruct: FreeSurfer 标签处理和逐结构 Dice 计算
"""

import math
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter

class AverageMeter(object):
    """计算并存储指标的均值、当前值和标准差。

    用法：
        meter = AverageMeter()
        meter.update(loss_value, batch_size)
        print(meter.avg)  # 累计均值
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def pad_image(img, target_size):
    """将图像填充到目标尺寸，不足部分用 0 补齐（右/下/后方向填充）。"""
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    slcs_to_pad = max(target_size[2] - img.shape[4], 0)
    padded_img = F.pad(img, (0, slcs_to_pad, 0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


class SpatialTransformer(nn.Module):
    """N 维空间变换器。

    根据位移场（displacement field）对源图像进行空间变形。
    内部维护一个与输入尺寸匹配的参考网格，加上位移场后得到采样坐标，
    再通过 grid_sample 完成双线性/最近邻插值采样。

    Args:
        size: 输入体数据的空间尺寸，如 (D, H, W)。
        mode: 插值模式，'bilinear'（双线性）或 'nearest'（最近邻）。
        device: 参考网格所在设备。
    """
    def __init__(self, size, mode='bilinear',device='cuda'):
        super().__init__()

        self.mode = mode
        self.dev = device
        # 创建参考采样网格（identity grid）
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)          # [ndims, *size]
        grid = torch.unsqueeze(grid, 0)    # [1, ndims, *size]
        grid = grid.type(torch.FloatTensor).to(self.dev)

        # 将网格注册为 buffer（不参与梯度计算，但会随模型移动到 GPU）
        # 注意：buffer 会包含在 state_dict 中，导致保存的模型文件较大
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        """对源图像 src 施加位移场 flow 进行空间变形。

        Args:
            src:  源图像 [B, C, D, H, W]。
            flow: 位移场 [B, 3, D, H, W]，单位为体素。
        Returns:
            变形后的图像 [B, C, D, H, W]。
        """
        # 将位移场叠加到参考网格上，得到新的采样坐标
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # 将体素坐标归一化到 [-1, 1]，满足 grid_sample 的要求
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # 将通道维移到最后，并反转坐标轴顺序（PyTorch grid_sample 要求 x,y,z 顺序）
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class register_model(nn.Module):
    """配准模型封装：将图像和位移场打包输入，输出变形后的图像。"""
    def __init__(self, img_size=(64, 256, 256), mode='bilinear',device='cuda'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode, device=device)
        self.dev = device
    def forward(self, x):
        img = x[0].to(self.dev)
        flow = x[1].to(self.dev)
        out = self.spatial_trans(img, flow)
        return out

def dice_val(y_pred, y_true, num_clus):
    """计算多类 Dice 分数（基于 one-hot 编码）。

    Args:
        y_pred: 预测分割标签 [B, 1, D, H, W]。
        y_true: 真实分割标签 [B, 1, D, H, W]。
        num_clus: 类别总数。
    Returns:
        所有类别的平均 Dice 分数（标量）。
    """
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))


def _dice_voi_batch(pred, true, voi_lbls):
    """批量计算指定感兴趣区域（VOI）标签的 Dice 分数。

    Args:
        pred: 预测标签 numpy 数组 [B, D, H, W]。
        true: 真实标签 numpy 数组 [B, D, H, W]。
        voi_lbls: 需要评估的标签值列表。
    Returns:
        所有样本和标签的平均 Dice 分数。
    """
    bs = pred.shape[0]
    dscs = np.zeros((bs, len(voi_lbls)), dtype=np.float64)
    for b in range(bs):
        pred_b = pred[b]
        true_b = true[b]
        for idx, lbl in enumerate(voi_lbls):
            pred_i = pred_b == lbl
            true_i = true_b == lbl
            intersection = np.sum(pred_i * true_i)
            union = np.sum(pred_i) + np.sum(true_i)
            dscs[b, idx] = (2. * intersection) / (union + 1e-5)
    return np.mean(dscs)


def dice_abdo(y_pred, y_true):
    """计算腹部 CT 数据集（AbdomenCTCT）的 Dice 分数，包含 13 个器官标签。"""
    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    pred = y_pred.detach().cpu().numpy()[:, 0, ...]
    true = y_true.detach().cpu().numpy()[:, 0, ...]
    return _dice_voi_batch(pred, true, VOI_lbls)

def dice_LPBA(y_pred, y_true):
    """计算 LPBA40 数据集的 Dice 分数，包含 54 个脑区标签。"""
    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54]
    pred = y_pred.detach().cpu().numpy()[:, 0, ...]
    true = y_true.detach().cpu().numpy()[:, 0, ...]
    return _dice_voi_batch(pred, true, VOI_lbls)

def dice_val_VOI(y_pred, y_true):
    """计算 IXI 数据集的 Dice 分数，包含 30 个感兴趣脑区标签。"""
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    pred = y_pred.detach().cpu().numpy()[:, 0, ...]
    true = y_true.detach().cpu().numpy()[:, 0, ...]
    return _dice_voi_batch(pred, true, VOI_lbls)

def jacobian_determinant_vxm(disp):
    """计算位移场的雅可比行列式。

    雅可比行列式用于衡量形变的局部体积变化：
      - |J| > 1 表示局部膨胀
      - |J| < 1 表示局部收缩
      - |J| < 0 表示折叠（不合理的形变）

    Args:
        disp: 位移场 [3, D, H, W]（3 个空间方向的位移分量）。
    Returns:
        雅可比行列式 [D, H, W]。
    """

    # 转置为 [D, H, W, 3] 格式
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # 构建参考网格并计算变形后坐标的梯度（即雅可比矩阵）
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))
    J = np.gradient(disp + grid)  # 对 (位移 + 参考网格) 求空间梯度

    # 3D 情况：通过行列式展开公式计算雅可比行列式
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # 3x3 行列式按第一行展开
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # 2D 情况
        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

import re
def process_label():
    """解析 FreeSurfer 的标签信息文件，构建标签索引到名称的映射字典。"""
    #process labeling information for FreeSurfer
    seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255]
    file1 = open('label_info.txt', 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict

def write2csv(line, name):
    """将一行文本追加写入 CSV 文件。"""
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def dice_val_substruct(y_pred, y_true, std_idx):
    """计算 46 个子结构的逐类 Dice 分数，返回逗号分隔的字符串（用于 CSV 输出）。"""
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=46)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(46):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line

