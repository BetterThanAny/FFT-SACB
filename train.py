
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

DEFAULT_BASE_DIR = '/root/autodl-tmp'


def parse_lp_ratio(value):
    parts = [float(x.strip()) for x in str(value).split(',') if x.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 4:
        return tuple(parts)
    raise argparse.ArgumentTypeError('lp_ratio must be one float or four comma-separated floats.')


def parse_weights(value):
    parts = [float(x.strip()) for x in str(value).split(',') if x.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('weights must be two comma-separated floats, e.g. 1,0.3')
    return parts


def format_lp_ratio(lp_ratio):
    if isinstance(lp_ratio, tuple):
        return '-'.join(f'{v:g}' for v in lp_ratio)
    return f'{lp_ratio:g}'


def build_parser():
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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_resume_ckpt(exp_dir, resume_path=''):
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
    base_dir = Path(args.base_dir).expanduser()

    g = torch.Generator()
    g.manual_seed(args.seed)
    setup_seed(seed=args.seed, cuda_deterministic=args.cuda_deterministic)

    lp_tag = format_lp_ratio(args.lp_ratio)
    tag = args.save_tag if args.save_tag else f'{args.dataset}_lp{lp_tag}'
    bs = args.batch_size
    weights = args.weights

    save_dir_name = f'sacb_ncc_{weights[0]}_reg_{weights[1]}_{tag}'
    exp_dir = Path('experiments') / save_dir_name
    log_dir = Path('logs') / save_dir_name
    csv_path = Path('csv') / f'sacb_ncc_{weights[0]}_reg_{weights[1]}_{tag}.csv'

    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_exists = csv_path.exists()
    csv_mode = 'a' if args.cont_training and csv_exists else 'w'
    with open(csv_path, csv_mode, newline='') as f:
        csvwriter = csv.writer(f)
        if csv_mode == 'w':
            csvwriter.writerow(['Index', 'Dice'])

    lr = args.lr
    epoch_start = args.epoch_start
    max_epoch = args.max_epoch
    cont_training = args.cont_training

    if cont_training and epoch_start == 0:
        epoch_start = args.resume_epoch
    if epoch_start < 0 or epoch_start >= max_epoch:
        raise ValueError(f'Effective epoch_start must be in [0, {max_epoch - 1}], got {epoch_start}')

    if args.dataset == 'ixi':
        atlas_dir = base_dir / 'IXI_data' / 'atlas.pkl'
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

    if args.dataset == 'lpba':
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

    if args.dataset == 'abd':
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

    model = SACB_Net(inshape=img_size, lp_ratio=args.lp_ratio)
    model.set_lp_ratio(args.lp_ratio)
    model.cuda()

    reg_model = utils.SpatialTransformer(size=img_size, mode='nearest').cuda()

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

    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]

    writer = SummaryWriter(log_dir=str(log_dir))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Num of params:', params)

    optimizer = torch.optim.Adam(model.parameters(), lr=updated_lr)

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        adjust_learning_rate(optimizer, epoch, max_epoch, lr)
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            output = model(x, y)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())

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

        eval_dsc = utils.AverageMeter()
        model.eval()
        reg_model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]

                _, flow = model(x, y)
                def_out = reg_model(x_seg.float(), flow)

                dsc = dice_score(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))

        with open(csv_path, 'a', newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow([epoch, eval_dsc.avg])

        save_checkpoint(
            {
                'state_dict': model.state_dict(),
            },
            save_dir=exp_dir,
            filename='dsc{:.4f}_e{}.pth.tar'.format(eval_dsc.avg, epoch),
        )

        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
    writer.close()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - epoch / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=20):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / filename
    torch.save(state, str(checkpoint_path))
    model_lists = sorted(save_dir.glob('*.pth.tar'), key=lambda p: p.stat().st_mtime)
    while len(model_lists) > max_model_num:
        model_lists[0].unlink()
        model_lists = sorted(save_dir.glob('*.pth.tar'), key=lambda p: p.stat().st_mtime)


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
