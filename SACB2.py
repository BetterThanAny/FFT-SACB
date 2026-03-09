"""
空间感知卷积块（SACB）—— K-Means 聚类版本。

这是 SACB 模块的原始实现，使用 GPU 上的 K-Means 聚类将体素按
特征相似性分组，然后为每组动态生成卷积核权重和偏置。

与 SACB1.py（FFT 频率分区版本）的区别：
  - SACB2 使用 K-Means 聚类（非确定性，依赖初始化）
  - SACB1 使用 FFT 频率分区（确定性，可微分）
  - SACB2 支持任意 num_k（聚类数），SACB1 固定 num_k=2
  - SACB2 的 MLP 输入是所有聚类中心的拼接，SACB1 的 MLP 输入是单个聚类中心

另外包含 cross_Sim 类（与 SACB1.py 中的版本功能相同）。
"""

# 注意：K-Means 版本显存占用较高
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from nn_util import get_act_layer, conv, unfoldNd
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, reduce
import numpy as np
from kmeans_gpu import KMeans
def tuple_(x, length = 1):
    """将标量 x 重复为指定长度的元组；若已是元组则直接返回。"""
    return x if isinstance(x, tuple) else ((x,) * length)


class KM_GPU():
    """基于 GPU 的 K-Means 聚类封装。

    用于将体素特征聚类为 num_k 个组，每组共享一套动态卷积核。

    Args:
        num_k: 聚类数量。
        rng_seed: 随机种子。
        tol: 收敛容差。
        m_iter: 最大迭代次数。
        fix_rng: 是否固定随机种子以保证可复现。
        max_neighbors: 最大邻居数限制。
    """
    def __init__(self,num_k=4, rng_seed=0,
                 tol=1e-9, m_iter=1e9, fix_rng=True,
                 max_neighbors=160*192*224):
        super(KM_GPU, self).__init__()   
        self.seed = rng_seed
        self.fix_rng = fix_rng
        
        self.km = KMeans(
            n_clusters= num_k,
            max_iter= int(m_iter),
 
            tolerance=tol,
            distance='euclidean',
            sub_sampling=None,
            max_neighbors=max_neighbors,
        )
    def set_k(self, k):
        """更新聚类数量。"""
        self.km.n_clusters = k

    def get_cluster_map(self, x):
        """对输入特征执行 K-Means 聚类。

        Args:
            x: 输入特征 [B, N, D_feat]，N 为体素数，D_feat 为特征维度。
        Returns:
            closest: 每个体素的聚类标签 [B, N]。
            centroid: 各聚类中心的特征向量 [B, K, D_feat]。
        """
        b,pts,feats = x.shape
        if self.fix_rng: np.random.seed(self.seed)
        if b==1:
            closest, centroid = self.km.fit_predict(x.squeeze(0))
            return closest.unsqueeze(0), centroid.unsqueeze(0)
        else:
            closests, centroids = [],[]
            for i in range(b):
                closest, centroid = self.km.fit_predict(x[i])
                closests.append(closest)
                centroids.append(centroid)
            closests = torch.stack(closests,  dim=0)
            centroids = torch.stack(centroids,  dim=0)
            return closests, centroids

class cross_Sim(nn.Module):
    """局部互相关注意力模块（K-Means 版本中的位移估计）。

    与 SACB1.py 中的 cross_Sim 功能相同：
    对固定图像特征的每个体素，在运动图像特征的局部窗口内计算注意力权重，
    加权求和邻域偏移向量，得到亚体素级位移估计 [B, 3, D, H, W]。

    Args:
        win_s: 局部窗口边长（默认 3，即 3x3x3 = 27 个邻居）。
    """
    def __init__(self, win_s=3):
        super(cross_Sim, self).__init__()
        self.wins = win_s
        self.win_len = win_s**3
              
    def forward(self, Fx, Fy, wins=None):
        """前向传播：计算运动特征 Fx 到固定特征 Fy 的局部位移场。"""
        if wins:
            self.wins = wins
            self.win_len = wins**3
        b, c, d, h, w = Fy.shape

        # 构建局部窗口内的相对偏移向量 [win^3, 3]
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [self.wins] * 3]
        grid = torch.stack(torch.meshgrid(vectors), -1).type(torch.FloatTensor)
        G = grid.reshape(self.win_len, 3).unsqueeze(0).unsqueeze(0).to(Fx.device)

        # 将固定特征展平为查询向量 [B, N, 1, C]
        Fy = rearrange(Fy, 'b c d h w -> b (d h w) 1 c')
        pd = self.wins // 2

        # 从运动特征中提取局部补丁 [B, N, win^3, C]
        Fx = F.pad(Fx,  tuple_(pd, length=6))
        Fx = Fx.unfold(2, self.wins, 1).unfold(3, self.wins, 1).unfold(4, self.wins, 1)
        Fx = rearrange(Fx, 'b c d h w wd wh ww -> b (d h w) (wd wh ww) c')

        # 注意力：query × keys -> softmax -> 加权求和偏移向量
        attn = (Fy @ Fx.transpose(-2, -1))
        sim = attn.softmax(dim=-1)
        out = (sim @ G)
        out = rearrange(out , 'b (d h w) 1 c -> b c d h w', d=d,h=h,w=w)
    
        return out



class SACB(nn.Module):
    """空间感知卷积块（K-Means 聚类版本）。

    核心流程：
      1. 输入投影：通过 1x1 卷积调整通道数
      2. 局部展开：将特征图展开为重叠的局部补丁
      3. 特征聚合：计算每个体素的特征摘要向量
      4. K-Means 聚类：将体素分为 num_k 组
      5. 动态卷积核生成：用所有聚类中心的拼接特征通过 MLP 生成权重和偏置
      6. 分组卷积：对每组体素应用对应的动态卷积核
      7. 残差连接：将输出加上输入（可选）

    Args:
        in_ch: 输入通道数。
        out_ch: 输出通道数。
        ks: 卷积核大小。
        in_proj_n: 输入投影倍率（1 = 保持通道数不变）。
        num_k: K-Means 聚类数。
        act: 激活函数。
        residual: 是否使用残差连接。
        mean_type: 特征聚合方式：
            's' = 对卷积核维度求均值（得到通道特征），
            'c' = 对通道维度求均值（得到核特征），
            其他 = 两者拼接。
        n_mlp: MLP 宽度倍率。
    """
    def __init__(self, in_ch, out_ch, ks, stride=1,
                 in_proj_n=1,
                 padding=1, dilation=1, groups=1,
                 num_k=4, 
                 act='prelu', residual=True, 
                 mean_type = 's',
                 scale_f=1,
                 n_mlp=1,
                 sample_n = 5,
                 m_iter= 1e10,
                 tol   = 1e-10,
                 fix_rng= False,
                 ):
        super(SACB, self).__init__()
        self.ks       = ks
        self.stride   = stride
        self.padding  =  tuple(x for x in reversed(_triple(padding)) for _ in range(2))
        self.dilation = _triple(dilation)
        self.num_k    = num_k
        self.res      = residual
        self.out_ch = out_ch
        in_ch_n = int(in_ch * in_proj_n)

        # 可学习的基础卷积权重 [out_ch, in_ch_n/groups, ks^3]
        self.w   = nn.Parameter(torch.Tensor(out_ch, in_ch_n // groups, self.ks**3))
        self.act = get_act_layer(act) if act else None
        self.reset_parameters()
        self.scale_f = scale_f
        self.mean_type = mean_type

        # GPU K-Means 聚类器
        self.km = KM_GPU(num_k=num_k, rng_seed=0, m_iter=m_iter, tol=tol, fix_rng=fix_rng)

        inner_dims = 128 * n_mlp
        inner_dims2 = 64 * n_mlp

        self.sample_n = sample_n

        # 根据 mean_type 确定 MLP 输入维度
        if   mean_type =='s':
            _in_c = in_ch_n
            self._in_c = _in_c
        elif mean_type =='c': _in_c = self.ks**3
        else: _in_c = in_ch + self.ks**3

        # MLP：所有聚类中心拼接 -> 动态卷积核权重调制因子
        self.get_kernel = nn.Sequential(
                nn.Linear(_in_c*num_k, inner_dims), nn.ReLU(),
                nn.Linear(inner_dims, inner_dims), nn.ReLU(),
                nn.Linear(inner_dims, self.ks**3*num_k), nn.Sigmoid()
                )

        # MLP：所有聚类中心拼接 -> 各组的通道偏置
        self.get_bias = nn.Sequential(
                nn.Linear(in_features=_in_c*num_k,  out_features=inner_dims2), nn.ReLU(),
                nn.Linear(in_features=inner_dims2, out_features=inner_dims2), nn.ReLU(),
                nn.Linear(in_features=inner_dims2, out_features=out_ch*num_k),
                )

        # 1x1 输入投影卷积
        self.proj_in  = conv(in_ch, in_ch_n, 1, 1, p=0, bias=False, act=None, norm=None)
      
    def reset_parameters(self):
        """使用 Kaiming 均匀分布初始化基础卷积权重。"""
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def set_num_k(self, k):
        """更新聚类数量。"""
        self.num_k = k
        self.km.set_k(k)

    def scale(self, x, factor, mode='nearest'):
        """对特征图进行空间缩放。"""
        if mode == 'nearest':
            return F.interpolate(x, scale_factor=factor, mode=mode)
        else:
            return F.interpolate(x, scale_factor=factor, mode='trilinear', align_corners=True)

    def feat_mean(self, x, mean_type='s'):
        """聚合展开后补丁特征为每体素的摘要向量。

        Args:
            x: 展开后的补丁 [B, C, nD, nH, nW, k1, k2, k3]。
            mean_type: 's' = 对核维度求均值，'c' = 对通道求均值，其他 = 两者拼接。
        """
        if   mean_type == 's': x = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) c', 'mean')
        elif mean_type == 'c': x = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) (k1 k2 k3)', 'mean')
        else:
            xs = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) c', 'mean')
            xc = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) (k1 k2 k3)', 'mean')
            x = torch.cat([xs, xc], -1)
        return x
       
    def forward(self, x, show_map=False):
        """应用基于 K-Means 聚类的动态卷积。

        Args:
            x: 输入特征图 [B, C, D, H, W]。
            show_map: 保留参数（未使用）。
        Returns:
            输出特征图 [B, out_ch, D, H, W]。
        """
        b,c,d,h,w = x.shape

        x_in = x                  # 保存输入用于残差连接
        x = self.proj_in(x)       # 1x1 投影卷积

        # 展开为重叠的局部补丁
        x_pad = F.pad(x, self.padding)
        x = x_pad.unfold(2,self.ks,self.stride).unfold(3,self.ks,self.stride).unfold(4,self.ks, self.stride)

        # 计算每体素的特征摘要并执行 K-Means 聚类
        x_mean = self.feat_mean(x, self.mean_type)
        cluster_idx, centroid = self.km.get_cluster_map(x_mean)

        # 重排补丁形状用于矩阵乘法 [1, B, C'*ks^3, N]
        x = rearrange(x,'b c nd nh nw k1 k2 k3 -> 1 b (c k1 k2 k3) (nd nh nw)')

        # 构建 one-hot 聚类掩码 [B, N, K]
        mask = F.one_hot(cluster_idx)

        # 用所有聚类中心的拼接特征生成动态卷积核权重和偏置
        w_i  = rearrange(self.get_kernel(centroid.flatten(1)), 'b (n k) -> n b 1 1 k', n=self.num_k, k=self.ks**3)
        weight =  w_i * self.w                                    # 逐元素调制基础权重
        weight =  rearrange(weight, 'n b o i k -> n b o (i k)')
        bias   = rearrange(self.get_bias(centroid.flatten(1)), 'b (n o) -> n b o 1', n=self.num_k, o=self.out_ch)

        # 分组卷积：每组体素乘以对应的动态权重和偏置，用掩码选择性叠加
        out = (weight@x + bias) * rearrange(mask, 'b l n -> n b 1 l').float()
        out = rearrange(out.sum(0), 'b o (d h w) -> b o d h w', d=d, h=h, w=w)

        if self.act: out = self.act(out)   # 激活函数
        if self.res: out = out + x_in      # 残差连接

        return out

