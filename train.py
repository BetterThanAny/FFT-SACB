"""
SACB-Net 训练脚本。

支持 IXI、LPBA、AbdomenCTCT 三个数据集的训练与验证。
主要流程：
  1. 解析命令行参数并进行校验
  2. 根据所选数据集构建训练集和验证集
  3. 初始化 SACB_Net 模型、损失函数（NCC + 梯度正则化）和优化器
  4. 执行训练循环，每个 epoch 结束后在验证集上计算 Dice 分数
  5. 保存模型检查点并记录训练日志
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import losses
import numpy as np
import torch
import utils
from dataset import datasets, trans
from natsort import natsorted
from model import SACB_Net
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 默认数据集根目录
DEFAULT_BASE_DIR = '/root/autodl-tmp'


def parse_lp_ratio(value):
    """解析低通滤波比率参数。

    支持两种格式：
      - 单个浮点数，如 "0.15" -> 所有尺度共享同一比率
      - 四个逗号分隔的浮点数，如 "0.1,0.15,0.2,0.25" -> 各尺度分别设置
    """
    parts = [float(x.strip()) for x in str(value).split(',') if x.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 4:
        return tuple(parts)
    raise argparse.ArgumentTypeError('lp_ratio must be one float or four comma-separated floats.')


def parse_weights(value):
    """解析损失权重参数，格式为两个逗号分隔的浮点数（图像相似性权重, 正则化权重）。"""
    parts = [float(x.strip()) for x in str(value).split(',') if x.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('weights must be two comma-separated floats, e.g. 1,0.3')
    return parts


def format_lp_ratio(lp_ratio):
    """将 lp_ratio 格式化为字符串，用于实验目录命名。"""
    if isinstance(lp_ratio, tuple):
        return '-'.join(f'{v:g}' for v in lp_ratio)
    return f'{lp_ratio:g}'


def build_parser():
    """构建命令行参数解析器，包含数据集选择、训练超参数、恢复训练等选项。"""
    parser = argparse.ArgumentParser(description='Train SACB-Net with FFT-based low/high-frequency partitioning.')
    parser.add_argument('--dataset', choices=['ixi', 'lpba', 'abd'], default='ixi')
    parser.add_argument('--lp-ratio', type=parse_lp_ratio, default=0.15,
                        help='Low-pass radius ratio. One value for all scales, or four values split by commas.')
    parser.add_argument('--weights', type=parse_weights, default=[1.0, 0.3],
                        help='Loss weights: image,regularization (e.g. 1,0.3).')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='Validation batch size. Use 0 to follow --batch-size.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--epoch-start', type=int, default=0)
    parser.add_argument('--cont-training', action='store_true')
    parser.add_argument('--resume-epoch', type=int, default=201,
                        help='Used only when --cont-training is enabled and --epoch-start is 0.')
    parser.add_argument('--resume-path', type=str, default='',
                        help='Optional checkpoint path for resuming training.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--base-dir', type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save-tag', type=str, default='',
                        help='Optional experiment tag. If empty, generated from dataset and lp_ratio.')
    parser.add_argument('--cuda-deterministic', action='store_true')
    return parser


def setup_seed(seed, cuda_deterministic=False):
    """设置全局随机种子，确保实验可复现。

    Args:
        seed: 随机种子值。
        cuda_deterministic: 为 True 时强制 cuDNN 确定性模式（牺牲速度换取可复现性）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    """DataLoader 的 worker 初始化函数，为每个 worker 设置独立随机种子。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_resume_ckpt(exp_dir, resume_path=''):
    """解析恢复训练所使用的检查点路径。

    如果指定了 resume_path 则直接使用；否则自动查找 exp_dir 下最新的 .pth.tar 文件。
    """
    if resume_path:
        p = Path(resume_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {p}')
        return p

    ckpts = natsorted(Path(exp_dir).glob('*.pth.tar'))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoint found in {exp_dir}')
    return ckpts[-1]


def validate_args(args):
    """校验命令行参数的合法性，不合法时抛出 ValueError 或 FileNotFoundError。"""
    if args.batch_size < 1:
        raise ValueError(f'--batch-size must be >= 1, got {args.batch_size}')
    if args.val_batch_size < 0:
        raise ValueError(f'--val-batch-size must be >= 0, got {args.val_batch_size}')
    if args.max_epoch <= 0:
        raise ValueError(f'--max-epoch must be > 0, got {args.max_epoch}')
    if args.epoch_start < 0 or args.epoch_start >= args.max_epoch:
        raise ValueError(f'--epoch-start must be in [0, {args.max_epoch - 1}], got {args.epoch_start}')
    if args.num_workers < 0:
        raise ValueError(f'--num-workers must be >= 0, got {args.num_workers}')
    if args.lr <= 0:
        raise ValueError(f'--lr must be > 0, got {args.lr}')
    if any(w < 0 for w in args.weights):
        raise ValueError(f'--weights must be non-negative, got {args.weights}')

    lp_values = args.lp_ratio if isinstance(args.lp_ratio, tuple) else (args.lp_ratio,)
    if any(v <= 0 for v in lp_values):
        raise ValueError(f'--lp-ratio values must be > 0, got {args.lp_ratio}')

    base_dir = Path(args.base_dir).expanduser()
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f'--base-dir does not exist or is not a directory: {base_dir}')


def main(args):
    """主训练函数。

    执行以下流程：
      1. 初始化随机种子和实验目录
      2. 根据数据集参数构建训练集/验证集 DataLoader
      3. 创建 SACB_Net 模型并加载检查点（如果是恢复训练）
      4. 定义损失函数：NCC（图像相似度）+ Grad3d（位移场平滑正则化）
      5. 训练循环：前向传播 -> 计算损失 -> 反向传播 -> 更新参数
      6. 每个 epoch 结束后在验证集上评估 Dice 分数并保存检查点
    """
    base_dir = Path(args.base_dir).expanduser()

    # --- 初始化随机种子 ---
    g = torch.Generator()
    g.manual_seed(args.seed)
    setup_seed(seed=args.seed, cuda_deterministic=args.cuda_deterministic)

    # --- 构建实验目录名称 ---
    lp_tag = format_lp_ratio(args.lp_ratio)
    tag = args.save_tag if args.save_tag else f'{args.dataset}_lp{lp_tag}'
    bs = args.batch_size
    weights = args.weights

    # 实验目录：模型检查点、TensorBoard 日志、CSV 记录
    save_dir_name = f'sacb_ncc_{weights[0]}_reg_{weights[1]}_{tag}'
    exp_dir = Path('experiments') / save_dir_name
    log_dir = Path('logs') / save_dir_name
    csv_path = Path('csv') / f'sacb_ncc_{weights[0]}_reg_{weights[1]}_{tag}.csv'

    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 初始化 CSV 记录文件 ---
    csv_exists = csv_path.exists()
    csv_mode = 'a' if args.cont_training and csv_exists else 'w'  # 恢复训练时追加写入
    with open(csv_path, csv_mode, newline='') as f:
        csvwriter = csv.writer(f)
        if csv_mode == 'w':
            csvwriter.writerow(['Index', 'Dice'])

    lr = args.lr
    epoch_start = args.epoch_start
    max_epoch = args.max_epoch
    cont_training = args.cont_training

    # --- 处理恢复训练的起始 epoch ---
    if cont_training and epoch_start == 0:
        epoch_start = args.resume_epoch
    if epoch_start < 0 or epoch_start >= max_epoch:
        raise ValueError(f'Effective epoch_start must be in [0, {max_epoch - 1}], got {epoch_start}')

    # --- 根据数据集类型配置数据路径、变换和评估函数 ---
    if args.dataset == 'ixi':
        atlas_dir = base_dir / 'IXI_data' / 'atlas.pkl'  # IXI 数据集使用图谱配准（atlas-to-subject）
        train_dir = base_dir / 'IXI_data' / 'Train'
        val_dir = base_dir / 'IXI_data' / 'Val'
        train_files = sorted(train_dir.glob('*.pkl'))
        val_files = sorted(val_dir.glob('*.pkl'))
        train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        val_composed = transforms.Compose([
            trans.Seg_norm(),
            trans.NumpyType((np.float32, np.int16))
        ])
        if not atlas_dir.is_file():
            raise FileNotFoundError(f'Atlas file not found: {atlas_dir}')
        train_set = datasets.IXIBrainDataset([str(p) for p in train_files], str(atlas_dir), transforms=train_composed)
        val_set = datasets.IXIBrainInferDataset([str(p) for p in val_files], str(atlas_dir), transforms=val_composed)
        dice_score = utils.dice_val_VOI
        img_size = (160, 192, 224)

    elif args.dataset == 'lpba':
        # LPBA 数据集使用受试者间配准（subject-to-subject）
        train_dir = base_dir / 'LPBA_data_2' / 'Train'
        val_dir = base_dir / 'LPBA_data_2' / 'Val'
        train_files = sorted(train_dir.glob('*.pkl'))
        val_files = sorted(val_dir.glob('*.pkl'))
        train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        val_composed = transforms.Compose([
            trans.Seg_norm2(),
            trans.NumpyType((np.float32, np.int16))
        ])
        train_set = datasets.LPBABrainDatasetS2S([str(p) for p in train_files], transforms=train_composed)
        val_set = datasets.LPBABrainInferDatasetS2S([str(p) for p in val_files], transforms=val_composed)
        dice_score = utils.dice_LPBA
        img_size = (160, 192, 160)

    elif args.dataset == 'abd':
        # 腹部 CT 数据集，同样使用 subject-to-subject 配准
        train_dir = base_dir / 'AbdomenCTCT' / 'Train'
        val_dir = base_dir / 'AbdomenCTCT' / 'Val'
        train_files = sorted(train_dir.glob('*.pkl'))
        val_files = sorted(val_dir.glob('*.pkl'))
        train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
        train_set = datasets.LPBABrainDatasetS2S([str(p) for p in train_files], transforms=train_composed)
        val_set = datasets.LPBABrainInferDatasetS2S([str(p) for p in val_files], transforms=val_composed)
        dice_score = utils.dice_abdo
        img_size = (192, 160, 224)

    if len(train_set) == 0:
        raise ValueError(f'No training samples found for dataset={args.dataset} under {train_dir}')
    if len(val_set) == 0:
        raise ValueError(f'No validation samples found for dataset={args.dataset} under {val_dir}')

    # --- 构建 DataLoader ---
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_bs = bs if args.val_batch_size == 0 else args.val_batch_size
    if args.val_batch_size == 0:
        print(f'[INFO] Validation batch_size follows train batch_size={bs}.')
    else:
        print(f'[INFO] Validation batch_size={val_bs}.')
    val_loader = DataLoader(
        val_set,
        batch_size=val_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- 初始化模型 ---
    model = SACB_Net(inshape=img_size, lp_ratio=args.lp_ratio)
    model.cuda()

    # 最近邻空间变换器，用于验证时对分割标签做变形
    reg_model = utils.SpatialTransformer(size=img_size, mode='nearest').cuda()

    # --- 恢复训练：加载检查点并调整学习率 ---
    if cont_training:
        model_dir = exp_dir
        updated_lr = round(lr * np.power(1 - epoch_start / max_epoch, 0.9), 8)
        ckpt_path = resolve_resume_ckpt(model_dir, args.resume_path)
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        if 'state_dict' not in ckpt:
            raise KeyError(f"Checkpoint missing 'state_dict': {ckpt_path}")
        model.load_state_dict(ckpt['state_dict'])
        print('Model: {} loaded!'.format(ckpt_path.name))
    else:
        updated_lr = lr

    # --- 定义损失函数 ---
    criterion = losses.NCC_vxm()        # 局部归一化互相关（图像相似度）
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]  # 位移场梯度 L2 正则化（鼓励平滑形变）

    writer = SummaryWriter(log_dir=str(log_dir))

    # --- 统计可训练参数量 ---
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Num of params:', params)

    # --- 初始化 Adam 优化器 ---
    optimizer = torch.optim.Adam(model.parameters(), lr=updated_lr)

    # ===== 训练主循环 =====
    best_dsc = 0.0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        adjust_learning_rate(optimizer, epoch, max_epoch, lr)  # 按 poly 策略衰减学习率
        loss_all = utils.AverageMeter()
        idx = 0

        # --- 训练阶段 ---
        for data in train_loader:
            idx += 1
            model.train()
            data = [t.cuda() for t in data]
            x = data[0]  # 运动图像（moving image）
            y = data[1]  # 固定图像（fixed image）
            output = model(x, y)  # output = (配准后图像, 位移场)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]  # 加权损失
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())

            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write(
                "\r" + 'Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(
                    idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()
                )
            )
            sys.stdout.flush()

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        # --- 验证阶段：计算 Dice 分数 ---
        eval_dsc = utils.AverageMeter()
        model.eval()
        reg_model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = [t.cuda() for t in data]
                x = data[0]      # 运动图像
                y = data[1]      # 固定图像
                x_seg = data[2]  # 运动图像的分割标签
                y_seg = data[3]  # 固定图像的分割标签

                _, flow = model(x, y)                    # 预测位移场
                def_out = reg_model(x_seg.float(), flow) # 用位移场变形分割标签

                dsc = dice_score(def_out.long(), y_seg.long())  # 计算变形后的 Dice 分数
                eval_dsc.update(dsc.item(), x.size(0))

        # 记录验证 Dice 到 CSV
        with open(csv_path, 'a', newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow([epoch, eval_dsc.avg])

        # 保存模型检查点
        save_checkpoint(
            {
                'state_dict': model.state_dict(),
            },
            save_dir=exp_dir,
            filename='dsc{:.4f}_e{}.pth.tar'.format(eval_dsc.avg, epoch),
        )

        # 保存最佳模型
        if eval_dsc.avg > best_dsc:
            best_dsc = eval_dsc.avg
            torch.save({'state_dict': model.state_dict()}, str(Path(exp_dir) / 'best_model.pth.tar'))
            print(f'Best model saved at epoch {epoch} with DSC={best_dsc:.4f}')

        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
    writer.close()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    """按多项式衰减策略调整学习率：lr = INIT_LR * (1 - epoch/MAX_EPOCHES)^power。"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - epoch / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=20):
    """保存模型检查点，并自动清理旧的检查点（最多保留 max_model_num 个）。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / filename
    torch.save(state, str(checkpoint_path))
    model_lists = sorted(
        [p for p in save_dir.glob('*.pth.tar') if p.name != 'best_model.pth.tar'],
        key=lambda p: p.stat().st_mtime,
    )
    while len(model_lists) > max_model_num:
        model_lists.pop(0).unlink()


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    GPU_num = torch.cuda.device_count()
    if not torch.cuda.is_available() or GPU_num == 0:
        raise RuntimeError('CUDA is not available. This training script currently requires GPU.')

    if args.gpu < 0 or args.gpu >= GPU_num:
        raise ValueError(f'--gpu must be in [0, {GPU_num - 1}], got {args.gpu}')

    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)

    torch.cuda.set_device(args.gpu)
    current_gpu = torch.cuda.current_device()
    print(f'Currently using GPU #{current_gpu}: ' + torch.cuda.get_device_name(current_gpu))

    main(args)
