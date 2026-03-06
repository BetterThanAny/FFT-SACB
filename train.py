
import argparse
import csv
import glob
import os
import random
import sys

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

DEFAULT_BASE_DIR = 'D:/'
os.environ.setdefault("base_dir", DEFAULT_BASE_DIR)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--epoch-start', type=int, default=0)
    parser.add_argument('--cont-training', action='store_true')
    parser.add_argument('--resume-epoch', type=int, default=201,
                        help='Used only when --cont-training is enabled and --epoch-start is 0.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--base-dir', type=str, default=os.getenv('base_dir', DEFAULT_BASE_DIR))
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


def main(args):
    os.environ['base_dir'] = args.base_dir

    g = torch.Generator()
    g.manual_seed(args.seed)
    setup_seed(seed=args.seed, cuda_deterministic=args.cuda_deterministic)

    lp_tag = format_lp_ratio(args.lp_ratio)
    tag = args.save_tag if args.save_tag else f'{args.dataset}_lp{lp_tag}'
    bs = args.batch_size
    weights = args.weights

    save_dir = 'sacb_ncc_{}_reg_{}_{}/'.format(weights[0], weights[1], tag)

    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)

    os.makedirs('./csv', exist_ok=True)
    csv_name = './csv/sacb_ncc_{}_reg_{}_{}.csv'.format(weights[0], weights[1], tag)

    f = open(csv_name, 'w')
    with f:
        fnames = ['Index', 'Dice']
        csvwriter = csv.DictWriter(f, fieldnames=fnames)
        csvwriter.writeheader()

    lr = args.lr
    epoch_start = args.epoch_start
    max_epoch = args.max_epoch
    cont_training = args.cont_training

    if cont_training and epoch_start == 0:
        epoch_start = args.resume_epoch

    if args.dataset == 'ixi':
        atlas_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/atlas.pkl')
        train_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/Train/')
        val_dir = os.path.join(os.getenv('base_dir'), 'IXI_data/Val/')
        train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        val_composed = transforms.Compose([
            trans.Seg_norm(),
            trans.NumpyType((np.float32, np.int16))
        ])
        train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
        val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
        dice_score = utils.dice_val_VOI
        img_size = (160, 192, 224)

    if args.dataset == 'lpba':
        train_dir = os.path.join(os.getenv('base_dir'), 'LPBA_data_2/Train/')
        val_dir = os.path.join(os.getenv('base_dir'), 'LPBA_data_2/Val/')
        train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        val_composed = transforms.Compose([
            trans.Seg_norm2(),
            trans.NumpyType((np.float32, np.int16))
        ])
        train_set = datasets.LPBABrainDatasetS2S(sorted(glob.glob(train_dir + '*.pkl')), transforms=train_composed)
        val_set = datasets.LPBABrainInferDatasetS2S(sorted(glob.glob(val_dir + '*.pkl')), transforms=val_composed)
        dice_score = utils.dice_LPBA
        img_size = (160, 192, 160)

    if args.dataset == 'abd':
        train_dir = os.path.join(os.getenv('base_dir'), 'AbdomenCTCT/Train/')
        val_dir = os.path.join(os.getenv('base_dir'), 'AbdomenCTCT/Val/')
        train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
        val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
        train_set = datasets.LPBABrainDatasetS2S(sorted(glob.glob(train_dir + '*.pkl')), transforms=train_composed)
        val_set = datasets.LPBABrainInferDatasetS2S(sorted(glob.glob(val_dir + '*.pkl')), transforms=val_composed)
        dice_score = utils.dice_abdo
        img_size = (192, 160, 224)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SACB_Net(inshape=img_size, lp_ratio=args.lp_ratio)
    model.set_lp_ratio(args.lp_ratio)
    model.cuda()

    reg_model = utils.SpatialTransformer(size=img_size, mode='nearest').cuda()

    if cont_training:
        model_dir = 'experiments/' + save_dir
        updated_lr = round(lr * np.power(1 - epoch_start / max_epoch, 0.9), 8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]

    writer = SummaryWriter(log_dir='logs/' + save_dir)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Num of params:', params)

    optimizer = torch.optim.Adam(model.parameters(), lr=updated_lr)

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
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
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]

                _, flow = model(x, y)
                def_out = reg_model(x_seg.float(), flow)

                dsc = dice_score(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))

        with open(csv_name, 'a') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow([epoch, eval_dsc.avg])

        save_checkpoint(
            {
                'state_dict': model.state_dict(),
            },
            save_dir='experiments/' + save_dir,
            filename='dsc{:.4f}_e{}.pth.tar'.format(eval_dsc.avg, epoch),
        )

        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
    writer.close()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - epoch / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=20):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    GPU_iden = args.gpu
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)

    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    main(args)
