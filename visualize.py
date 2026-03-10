"""
Visualization script for SACB-Net results.

Generates comparison figures: fixed/moving/warped images, difference heatmaps,
segmentation overlays, and deformation grid plots.

Usage:
    python visualize.py --input results/pair_0000.npz
    python visualize.py --input results/pair_0000.npz --views axial coronal --slice_idx 80
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='SACB-Net Visualization')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to .npz result file from infer.py')
    parser.add_argument('--output_dir', type=str, default='vis/',
                        help='Directory to save visualization PNGs (default: vis/)')
    parser.add_argument('--slice_idx', type=int, default=None,
                        help='Slice index (default: middle slice)')
    parser.add_argument('--views', type=str, nargs='+',
                        default=['axial', 'sagittal', 'coronal'],
                        choices=['axial', 'sagittal', 'coronal'],
                        help='Views to visualize (default: all three)')
    parser.add_argument('--grid_spacing', type=int, default=4,
                        help='Grid line spacing for deformation grid (default: 4)')
    return parser.parse_args()


def get_slice(volume, view, idx=None):
    """Extract a 2D slice from a 3D volume.

    Args:
        volume: 3D numpy array (D, H, W)
        view: 'axial', 'sagittal', or 'coronal'
        idx: slice index (if None, use middle)

    Returns:
        2D numpy array
    """
    if view == 'axial':
        if idx is None:
            idx = volume.shape[0] // 2
        return volume[idx, :, :]
    elif view == 'sagittal':
        if idx is None:
            idx = volume.shape[2] // 2
        return volume[:, :, idx]
    elif view == 'coronal':
        if idx is None:
            idx = volume.shape[1] // 2
        return volume[:, idx, :]


def plot_image_comparison(fixed, moving, warped, view, slice_idx, save_path):
    """Plot fixed / moving / warped side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ['Fixed', 'Moving', 'Warped']
    images = [
        get_slice(fixed, view, slice_idx),
        get_slice(moving, view, slice_idx),
        get_slice(warped, view, slice_idx),
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.T, cmap='gray', origin='lower')
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    fig.suptitle(f'{view.capitalize()} View', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_difference_map(fixed, warped, view, slice_idx, save_path):
    """Plot |fixed - warped| as a heatmap alongside the images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    fixed_slice = get_slice(fixed, view, slice_idx)
    diff_slice = get_slice(np.abs(fixed - warped), view, slice_idx)
    warped_slice = get_slice(warped, view, slice_idx)

    axes[0].imshow(fixed_slice.T, cmap='gray', origin='lower')
    axes[0].set_title('Fixed', fontsize=14)
    axes[0].axis('off')

    im = axes[1].imshow(diff_slice.T, cmap='hot', origin='lower')
    axes[1].set_title('|Fixed - Warped|', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(warped_slice.T, cmap='gray', origin='lower')
    axes[2].set_title('Warped', fontsize=14)
    axes[2].axis('off')

    fig.suptitle(f'Difference Map - {view.capitalize()} View', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_segmentation_overlay(fixed_seg, warped_seg, view, slice_idx, save_path):
    """Plot fixed segmentation vs warped segmentation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    fixed_slice = get_slice(fixed_seg, view, slice_idx)
    warped_slice = get_slice(warped_seg, view, slice_idx)

    num_labels = max(int(fixed_seg.max()), int(warped_seg.max())) + 1
    cmap = plt.cm.get_cmap('nipy_spectral', num_labels)

    axes[0].imshow(fixed_slice.T, cmap=cmap, origin='lower',
                   interpolation='nearest', vmin=0, vmax=num_labels)
    axes[0].set_title('Fixed Segmentation', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(warped_slice.T, cmap=cmap, origin='lower',
                   interpolation='nearest', vmin=0, vmax=num_labels)
    axes[1].set_title('Warped Segmentation', fontsize=14)
    axes[1].axis('off')

    fig.suptitle(f'Segmentation - {view.capitalize()} View', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_deformation_grid(flow, view, slice_idx, save_path, grid_spacing=4):
    """Plot deformation grid visualization.

    Args:
        flow: (3, D, H, W) flow field
        view: 'axial', 'sagittal', or 'coronal'
        slice_idx: slice index (None for middle)
        save_path: output file path
        grid_spacing: spacing between grid lines
    """
    if view == 'axial':
        if slice_idx is None:
            slice_idx = flow.shape[1] // 2
        u = flow[1, slice_idx, :, :]  # H displacement
        v = flow[2, slice_idx, :, :]  # W displacement
        shape = (flow.shape[2], flow.shape[3])
    elif view == 'sagittal':
        if slice_idx is None:
            slice_idx = flow.shape[3] // 2
        u = flow[0, :, :, slice_idx]  # D displacement
        v = flow[1, :, :, slice_idx]  # H displacement
        shape = (flow.shape[1], flow.shape[2])
    elif view == 'coronal':
        if slice_idx is None:
            slice_idx = flow.shape[2] // 2
        u = flow[0, :, slice_idx, :]  # D displacement
        v = flow[2, :, slice_idx, :]  # W displacement
        shape = (flow.shape[1], flow.shape[3])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Create grid coordinates
    xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # Add displacements to get deformed grid
    xx_def = xx + v
    yy_def = yy + u

    # Draw deformed grid lines
    for i in range(0, shape[0], grid_spacing):
        ax.plot(xx_def[i, :], yy_def[i, :], 'b-', linewidth=0.5, alpha=0.7)
    for j in range(0, shape[1], grid_spacing):
        ax.plot(xx_def[:, j], yy_def[:, j], 'b-', linewidth=0.5, alpha=0.7)

    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    ax.set_aspect('equal')
    ax.set_title(f'Deformation Grid - {view.capitalize()} View', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = get_args()

    # Load data
    data = np.load(args.input)
    fixed = data['fixed']
    moving = data['moving']
    warped = data['warped']
    flow = data['flow']

    has_seg = 'fixed_seg' in data and 'warped_seg' in data
    if has_seg:
        fixed_seg = data['fixed_seg']
        warped_seg = data['warped_seg']

    os.makedirs(args.output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(args.input))[0]

    for view in args.views:
        slice_idx = args.slice_idx  # None means middle

        # 1. Image comparison: fixed / moving / warped
        save_path = os.path.join(args.output_dir, f'{basename}_{view}_comparison.png')
        plot_image_comparison(fixed, moving, warped, view, slice_idx, save_path)
        print(f'Saved: {save_path}')

        # 2. Difference heatmap: |fixed - warped|
        save_path = os.path.join(args.output_dir, f'{basename}_{view}_difference.png')
        plot_difference_map(fixed, warped, view, slice_idx, save_path)
        print(f'Saved: {save_path}')

        # 3. Segmentation comparison
        if has_seg:
            save_path = os.path.join(args.output_dir, f'{basename}_{view}_segmentation.png')
            plot_segmentation_overlay(fixed_seg, warped_seg, view, slice_idx, save_path)
            print(f'Saved: {save_path}')

        # 4. Deformation grid
        save_path = os.path.join(args.output_dir, f'{basename}_{view}_grid.png')
        plot_deformation_grid(flow, view, slice_idx, save_path, args.grid_spacing)
        print(f'Saved: {save_path}')

    print(f'\nAll visualizations saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
