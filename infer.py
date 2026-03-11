"""
Inference script for FFT-SACB.

Loads a trained checkpoint and evaluates on the test/validation set.
Computes Dice scores and Jacobian determinant statistics.

Usage:
    python infer.py --checkpoint experiments/xxx.pth.tar --dataset ixi
    python infer.py --checkpoint experiments/xxx.pth.tar --dataset lpba --save_results
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from model import SACB_Net
from dataset import datasets, trans


def parse_lp_ratio(value):
    parts = [float(x.strip()) for x in str(value).split(',') if x.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 4:
        return tuple(parts)
    raise argparse.ArgumentTypeError('lp_ratio must be one float or four comma-separated floats.')


def get_args():
    parser = argparse.ArgumentParser(description='FFT-SACB Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth.tar)')
    parser.add_argument('--dataset', type=str, default='ixi',
                        choices=['ixi', 'lpba', 'abd'],
                        help='Dataset to evaluate on (default: ixi)')
    parser.add_argument('--base-dir', type=str, default='/root/autodl-tmp',
                        help='Base data directory')
    parser.add_argument('--lp-ratio', type=parse_lp_ratio, default=0.15,
                        help='Low-pass radius ratio (must match training config)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--save-dir', type=str, default='results/',
                        help='Directory to save inference results (default: results/)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save warped images and flow fields as .npz')
    return parser.parse_args()


def compute_jacobian_stats(flow):
    """Compute Jacobian determinant statistics from a flow field."""
    jac_det = utils.jacobian_determinant_vxm(flow)
    num_neg = np.sum(jac_det <= 0)
    total = np.prod(jac_det.shape)
    return {
        'percent_neg_jac': num_neg / total * 100,
        'num_neg': int(num_neg),
        'total': int(total),
    }


def main():
    args = get_args()

    torch.cuda.set_device(args.gpu)
    print(f'Using GPU: {torch.cuda.get_device_name(args.gpu)}')

    base_dir = Path(args.base_dir).expanduser()

    # Dataset config
    if args.dataset == 'ixi':
        atlas_dir = str(base_dir / 'IXI_data' / 'atlas.pkl')
        val_dir = base_dir / 'IXI_data' / 'Val'
        val_composed = transforms.Compose([
            trans.Seg_norm(),
            trans.NumpyType((np.float32, np.int16))
        ])
        val_set = datasets.IXIBrainInferDataset(
            [str(p) for p in sorted(val_dir.glob('*.pkl'))],
            atlas_dir, transforms=val_composed)
        dice_score = utils.dice_val_VOI
        img_size = (160, 192, 224)
    elif args.dataset == 'lpba':
        val_dir = base_dir / 'LPBA_data_2' / 'Val'
        val_composed = transforms.Compose([
            trans.Seg_norm2(),
            trans.NumpyType((np.float32, np.int16))
        ])
        val_set = datasets.LPBABrainInferDatasetS2S(
            [str(p) for p in sorted(val_dir.glob('*.pkl'))],
            transforms=val_composed)
        dice_score = utils.dice_LPBA
        img_size = (160, 192, 160)
    elif args.dataset == 'abd':
        val_dir = base_dir / 'AbdomenCTCT' / 'Val'
        val_composed = transforms.Compose([
            trans.NumpyType((np.float32, np.int16))
        ])
        val_set = datasets.LPBABrainInferDatasetS2S(
            [str(p) for p in sorted(val_dir.glob('*.pkl'))],
            transforms=val_composed)
        dice_score = utils.dice_abdo
        img_size = (192, 160, 224)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)

    if len(val_set) == 0:
        raise ValueError(f'No validation samples found for dataset={args.dataset} under {val_dir}')

    # Initialize model (FFT-SACB: num_k=2, configurable lp_ratio)
    model = SACB_Net(inshape=img_size, lp_ratio=args.lp_ratio)
    model.cuda()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.gpu}')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f'Checkpoint loaded: {args.checkpoint}')

    reg_model = utils.SpatialTransformer(size=img_size, mode='nearest').cuda()

    if args.save_results:
        os.makedirs(args.save_dir, exist_ok=True)

    dice_scores = []
    jac_stats_list = []

    print(f'\nEvaluating on {args.dataset} dataset ({len(val_set)} pairs)...\n')

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_warped, flow = model(x, y)
            def_out = reg_model(x_seg.float(), flow)

            dsc = dice_score(def_out.long(), y_seg.long())
            dice_scores.append(dsc)

            flow_np = flow.detach().cpu().numpy()[0]
            jac_info = compute_jacobian_stats(flow_np)
            jac_stats_list.append(jac_info)

            print(f'[{idx + 1:3d}/{len(val_loader)}] '
                  f'Dice: {dsc:.4f}  |  '
                  f'%|J|<=0: {jac_info["percent_neg_jac"]:.3f}%')

            if args.save_results:
                np.savez_compressed(
                    os.path.join(args.save_dir, f'pair_{idx:04d}.npz'),
                    moving=x.cpu().numpy()[0, 0],
                    fixed=y.cpu().numpy()[0, 0],
                    warped=x_warped.cpu().numpy()[0, 0],
                    flow=flow_np,
                    moving_seg=x_seg.cpu().numpy()[0, 0],
                    fixed_seg=y_seg.cpu().numpy()[0, 0],
                    warped_seg=def_out.cpu().numpy()[0, 0],
                )

    dice_arr = np.array(dice_scores)
    jac_neg_arr = np.array([s['percent_neg_jac'] for s in jac_stats_list])

    print('\n' + '=' * 60)
    print(f'Dataset:    {args.dataset}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Num pairs:  {len(dice_scores)}')
    print('-' * 60)
    print(f'Dice:       {dice_arr.mean():.4f} +/- {dice_arr.std():.4f}')
    print(f'%|J|<=0:    {jac_neg_arr.mean():.4f} +/- {jac_neg_arr.std():.4f}')
    print('=' * 60)

    if args.save_results:
        print(f'\nResults saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
