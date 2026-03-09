# FFT-SACB: 基于傅里叶频域分区的空间感知卷积医学图像配准系统

基于 [SACB-Net (CVPR 2025)](https://arxiv.org/abs/2503.19592) 进行算法改进与工程化重构。将原始 K-Means 聚类分区替换为 FFT 频域分区，并构建了完整的训练、测试与实验自动化体系。

## 改进动机

原始 SACB-Net 的空间分区模块基于 GPU K-Means 聚类，存在以下问题：

| 问题 | 说明 |
|------|------|
| 非确定性 | 随机初始化导致实验结果不可复现 |
| 收敛慢 | 迭代聚类计算开销大 |
| 梯度不可传播 | 聚类分配过程不可微 |

本项目用 **3D FFT 频域分区** 替代 K-Means，实现确定性、可微、高效的空间感知卷积。

## 核心改进

### 1. FFT 频域分区（`SACB1.py`）

- **FrequencyPartition 模块**：通过 3D FFT 将特征图变换到频域，以可配置半径（`lp_ratio`）的球形低通掩码将空间位置划分为低频主导区与高频主导区
- 相比 K-Means（`SACB2.py`）：完全确定性、支持梯度传播、具有频率物理含义
- FFT 掩码缓存机制，避免重复计算
- cuFFT → CPU 自动回退策略，兼顾性能与跨设备兼容性
- 支持多尺度独立 `lp_ratio`（4 个解码层各自设定频率截止参数）

### 2. 训练系统重构（`train.py`）

- 统一训练接口：支持 IXI / LPBA40 / AbdomenCTCT 三个数据集
- 多格式参数解析：单值/逐尺度 `lp_ratio`、多权重配比
- 参数合法性预校验，防止训练中途崩溃
- 断点续训：checkpoint resume + 学习率自动衔接
- 可复现性保障：统一种子管理（torch/numpy/random）、DataLoader worker 种子控制、cuDNN 确定性模式

### 3. 测试与质量保障（`tests/`）

- 4 个测试模块（243 行）：FFT 分区正确性、参数解析、S2S 配对逻辑、损失函数设备兼容性与梯度流
- 修复原代码 `meshgrid` 缺少 `indexing='ij'`、`Variable()` 废弃包装等问题
- 端到端烟雾测试（`smoke_forward_test.py`）

### 4. 实验自动化（`scripts/`）

- 标准训练启动器（`run_base.sh`）
- 1-epoch 烟雾测试（`run_smoke.sh`）
- 多维超参数网格搜索（`lp_ratio × weights × lr × seed`），自动汇总结果

## 项目结构

```
SACB_Net/
├── model.py            # 多尺度编码器-解码器网络（支持逐尺度 lp_ratio）
├── SACB1.py            # SACB 模块 — FFT 频域分区（本项目核心改进）
├── SACB2.py            # SACB 模块 — K-Means 聚类（原始基线，用于对比）
├── nn_util.py          # 网络组件（STN、卷积块等）
├── train.py            # 训练脚本（参数校验、断点续训、种子管理）
├── losses.py           # 损失函数库（NCC、SSIM、MIND-SSC、MI、Grad）
├── utils.py            # 空间变换器、Dice 指标、Jacobian 行列式
├── dataset/
│   ├── datasets.py     # 数据集类（IXI、LPBA、AbdomenCTCT）
│   ├── trans.py        # 数据增强与预处理
│   └── data_utils.py   # Pickle 加载工具
├── scripts/
│   ├── run_base.sh     # 标准训练启动
│   ├── run_smoke.sh    # 烟雾测试
│   └── sweep_hparams.sh # 超参数搜索
├── tests/
│   ├── test_sacb_fft.py       # FFT 分区单元测试
│   ├── test_train_args.py     # 参数解析测试
│   ├── test_dataset_pairs.py  # 数据集配对测试
│   └── test_losses_device.py  # 损失函数设备兼容测试
├── smoke_forward_test.py      # 端到端前向传播测试
└── requirements.txt
```

## 环境配置

```bash
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```

**主要依赖：** PyTorch 1.13.1+, MONAI 1.4.0, einops, timm 0.9.2, pystrum

## 数据集

| 数据集 | 配准类型 | 图像尺寸 | 评估标签 |
|--------|---------|----------|---------|
| [IXI](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) | Atlas-to-subject | 160 x 192 x 224 | 30 脑区 ROI |
| [LPBA](https://loni.usc.edu/research/atlases) | Subject-to-subject | 160 x 192 x 160 | 54 脑区 |
| [Abdomen CT-CT](https://learn2reg.grand-challenge.org/Datasets/) | Subject-to-subject | 192 x 160 x 224 | 13 器官 |

感谢 [@Junyu Chen](https://github.com/junyuchen245) 提供预处理后的 IXI 数据。

数据目录结构：

```
<base-dir>/
├── IXI_data/
│   ├── atlas.pkl
│   ├── Train/*.pkl
│   └── Val/*.pkl
├── LPBA_data_2/
│   ├── Train/*.pkl
│   └── Val/*.pkl
└── AbdomenCTCT/
    ├── Train/*.pkl
    └── Val/*.pkl
```

## 训练

```bash
python train.py \
    --dataset ixi \
    --base-dir /root/autodl-tmp \
    --lp-ratio 0.15 \
    --weights 1,0.3 \
    --max-epoch 300 \
    --batch-size 1 \
    --lr 1e-4 \
    --gpu 0
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `ixi` | 数据集：`ixi`、`lpba` 或 `abd` |
| `--base-dir` | `/root/autodl-tmp` | 数据集根目录 |
| `--lp-ratio` | `0.15` | FFT 低通半径比。单个浮点数（所有尺度共享）或四个逗号分隔值（逐尺度） |
| `--weights` | `1,0.3` | 损失权重：`<NCC>,<正则化>` |
| `--batch-size` | `1` | 训练批大小 |
| `--val-batch-size` | `1` | 验证批大小（0 = 跟随训练批大小） |
| `--lr` | `1e-4` | 初始学习率（多项式衰减） |
| `--max-epoch` | `300` | 总训练轮数 |
| `--epoch-start` | `0` | 起始轮次 |
| `--cont-training` | off | 从 checkpoint 续训 |
| `--resume-path` | `""` | 指定 checkpoint 路径 |
| `--seed` | `0` | 随机种子 |
| `--num-workers` | `8` | DataLoader 工作进程数 |
| `--gpu` | `0` | GPU 编号 |
| `--save-tag` | `""` | 实验标签（为空时自动生成） |

**损失函数：** NCC（归一化互相关）+ 位移场 L2 梯度正则化

**学习率调度：** 多项式衰减 `lr * (1 - epoch/max_epoch)^0.9`

检查点、TensorBoard 日志和 CSV 指标分别保存在 `experiments/`、`logs/` 和 `csv/` 目录下。

## 关于多 GPU 训练

本项目仅支持**单 GPU 训练**。3D 医学图像配准使用大尺寸单体积（如 160x192x224），典型 batch size 为 1，无法通过 DataParallel 拆分。多 GPU 环境下建议并行运行独立实验：

```bash
# 终端 1
CUDA_VISIBLE_DEVICES=0 python train.py --gpu 0 ...

# 终端 2
CUDA_VISIBLE_DEVICES=1 python train.py --gpu 0 ...
```

## 原始论文

本项目基于以下工作进行改进：

```bibtex
@InProceedings{Cheng_2025_CVPR,
    author    = {Cheng, Xinxing and Zhang, Tianyang and Lu, Wenqi and Meng, Qingjie and Frangi, Alejandro F. and Duan, Jinming},
    title     = {SACB-Net: Spatial-awareness Convolutions for Medical Image Registration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5227-5237}
}
```

## 致谢

感谢 [SACB-Net](https://github.com/)、[ModeT](https://github.com/ZAX130/SmileCode)、[CANNet](https://github.com/Duanyll/CANConv) 和 [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) 项目。
