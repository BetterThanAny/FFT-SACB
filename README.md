# SACB-Net: Spatial-awareness Convolutions for Medical Image Registration

The official implementation of SACB-Net [![CVPR](https://img.shields.io/badge/CVPR2025-68BC71.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_SACB-Net_Spatial-awareness_Convolutions_for_Medical_Image_Registration_CVPR_2025_paper.html)  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.19592)

## Overview

SACB-Net is a 3D medical image registration network built on **Spatial-Awareness Convolution Blocks (SACB)**. The key idea is to partition spatial locations by their frequency characteristics via 3D FFT (low-frequency dominant vs. high-frequency dominant) and apply dynamically generated convolution kernels tailored to each partition. The network adopts a coarse-to-fine architecture with a shared 5-level feature pyramid encoder and a multi-scale decoder that combines SACB modules with cross-similarity based displacement estimation.

## Project Structure

```
SACB_Net/
├── model.py            # SACB_Net: encoder + multi-scale decoder
├── SACB1.py            # SACB module (FFT frequency partition, default)
├── SACB2.py            # SACB module (K-Means clustering, alternative)
├── nn_util.py          # Neural network utilities (STN, conv blocks, etc.)
├── train.py            # Training script
├── losses.py           # Loss functions (NCC, SSIM, MIND-SSC, MI, Grad)
├── utils.py            # Spatial transformer, Dice metrics, Jacobian
├── dataset/
│   ├── datasets.py     # Dataset classes (IXI, LPBA, AbdomenCTCT)
│   ├── trans.py        # Data augmentation transforms
│   └── data_utils.py   # Pickle loading and utilities
├── scripts/            # Shell scripts for training
├── tests/              # Unit tests
└── requirements.txt
```

## Environment Setup

```bash
conda create -n sacbnet python=3.9
conda activate sacbnet
pip install -r requirements.txt
```

**Key dependencies:** PyTorch 1.13.1, MONAI 1.4.0, einops, timm 0.9.2, pystrum

## Dataset

| Dataset | Registration Type | Image Size | Eval Labels |
|---------|------------------|------------|-------------|
| [IXI](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) | Atlas-to-subject | 160 x 192 x 224 | 30 brain ROIs |
| [LPBA](https://loni.usc.edu/research/atlases) | Subject-to-subject | 160 x 192 x 160 | 54 brain regions |
| [Abdomen CT-CT](https://learn2reg.grand-challenge.org/Datasets/) | Subject-to-subject | 192 x 160 x 224 | 13 organs |

Thanks [@Junyu Chen](https://github.com/junyuchen245) for the preprocessed IXI data.

Place the data under a base directory (default `/root/autodl-tmp`):

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

## Training

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

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ixi` | Dataset: `ixi`, `lpba`, or `abd` |
| `--base-dir` | `/root/autodl-tmp` | Root directory containing dataset folders |
| `--lp-ratio` | `0.15` | FFT low-pass radius ratio. Single float (shared) or four comma-separated floats (per-scale) |
| `--weights` | `1,0.3` | Loss weights: `<NCC>,<regularization>` |
| `--batch-size` | `1` | Training batch size |
| `--val-batch-size` | `1` | Validation batch size (0 = follow training batch size) |
| `--lr` | `1e-4` | Initial learning rate (poly decay) |
| `--max-epoch` | `300` | Total training epochs |
| `--epoch-start` | `0` | Starting epoch |
| `--cont-training` | off | Resume training from checkpoint |
| `--resume-path` | `""` | Specific checkpoint path for resuming |
| `--seed` | `0` | Random seed |
| `--num-workers` | `8` | DataLoader workers |
| `--gpu` | `0` | GPU index |
| `--save-tag` | `""` | Experiment tag (auto-generated if empty) |

**Loss function:** NCC (normalized cross-correlation) + L2 gradient regularization on the displacement field.

**Learning rate schedule:** polynomial decay `lr * (1 - epoch/max_epoch)^0.9`.

Checkpoints, TensorBoard logs, and CSV metrics are saved under `experiments/`, `logs/`, and `csv/` respectively.

## Pre-trained Weights

[Google Drive](https://drive.google.com/drive/folders/1XW19iuyCyg3YGmCpLFGGFjdPFi73xxwh?usp=share_link)

## Citation

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

## Acknowledgments

We sincerely acknowledge the [ModeT](https://github.com/ZAX130/SmileCode), [CANNet](https://github.com/Duanyll/CANConv) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) projects.
