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
    return x if isinstance(x, tuple) else ((x,) * length)


class FrequencyPartition(nn.Module):
    """Split 3D feature maps into low/high-frequency dominant regions via FFT."""

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
        return (high_energy > low_energy).long().view(b, -1)

    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            cluster_idx: [B, D*H*W], 0=low-dominant, 1=high-dominant
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
    def __init__(self, win_s=3):
        super(cross_Sim, self).__init__()
        self.wins = win_s
        self.win_len = win_s ** 3

    def forward(self, Fx, Fy, wins=None):
        if wins:
            self.wins = wins
            self.win_len = wins ** 3
        b, c, d, h, w = Fy.shape

        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [self.wins] * 3]
        grid = torch.stack(torch.meshgrid(vectors, indexing='ij'), -1).type(torch.FloatTensor)

        G = grid.reshape(self.win_len, 3).unsqueeze(0).unsqueeze(0).to(Fx.device)

        Fy = rearrange(Fy, 'b c d h w -> b (d h w) 1 c')
        pd = self.wins // 2

        Fx = F.pad(Fx, tuple_(pd, length=6))
        Fx = Fx.unfold(2, self.wins, 1).unfold(3, self.wins, 1).unfold(4, self.wins, 1)
        Fx = rearrange(Fx, 'b c d h w wd wh ww -> b (d h w) (wd wh ww) c')

        attn = Fy @ Fx.transpose(-2, -1)
        sim = attn.softmax(dim=-1)
        out = sim @ G
        out = rearrange(out, 'b (d h w) 1 c -> b c d h w', d=d, h=h, w=w)

        return out


class SACB(nn.Module):
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
        num_k=4,
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
        self.num_k = 2
        self.res = residual
        self.out_ch = out_ch
        in_ch_n = int(in_ch * in_proj_n)

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

        self.get_kernel = nn.Sequential(
            nn.Linear(_in_c, inner_dims),
            nn.ReLU(),
            nn.Linear(inner_dims, inner_dims),
            nn.ReLU(),
            nn.Linear(inner_dims, self.ks ** 3),
            nn.Sigmoid(),
        )

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
        # Keep compatibility with existing callers; FFT version is fixed to 2 groups.
        self.num_k = 2

    def set_lp_ratio(self, lp_ratio):
        self.freq_partition.set_lp_ratio(lp_ratio)

    def set_force_cpu_fft(self, force_cpu_fft):
        self.freq_partition.set_force_cpu_fft(force_cpu_fft)

    def scale(self, x, factor, mode='nearest'):
        if mode == 'nearest':
            return F.interpolate(x, scale_factor=factor, mode=mode)
        return F.interpolate(x, scale_factor=factor, mode='trilinear', align_corners=True)

    def feat_mean(self, x, mean_type='s'):
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
        b, c, d, h, w = x.shape
        x_in = x
        x = self.proj_in(x)

        x_pad = F.pad(x, self.padding)
        x_unfold = x_pad.unfold(2, self.ks, self.stride).unfold(3, self.ks, self.stride).unfold(4, self.ks, self.stride)

        x_mean = self.feat_mean(x_unfold, self.mean_type)
        cluster_idx = self.freq_partition(x)

        # Derive two frequency centroids in the same feature space as original k-means centroids.
        low_mask = cluster_idx.eq(0).float()
        high_mask = cluster_idx.eq(1).float()
        low_denom = low_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        high_denom = high_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        low_centroid = (x_mean * low_mask.unsqueeze(-1)).sum(dim=1) / low_denom
        high_centroid = (x_mean * high_mask.unsqueeze(-1)).sum(dim=1) / high_denom
        centroids = [low_centroid, high_centroid]

        x = rearrange(x_unfold, 'b c nd nh nw k1 k2 k3 -> b (c k1 k2 k3) (nd nh nw)')
        out = x.new_zeros(b, self.out_ch, d * h * w)

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
