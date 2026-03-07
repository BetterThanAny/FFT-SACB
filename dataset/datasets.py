import os, glob
import torch, sys
from torch.utils.data import Dataset
import pickle
import numpy as np
    
def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms
        self.atlas_img, self.atlas_seg = pkload(self.atlas_path)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x = self.atlas_img.copy()
        y, y_seg = pkload(path)

        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms
        self.atlas_img, self.atlas_seg = pkload(self.atlas_path)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x = self.atlas_img.copy()
        x_seg = self.atlas_seg.copy()
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
    

class OASISBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
    

class LPBABrainDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms
        self.num_paths = len(self.paths)

    def _pair_from_index(self, index):
        n = self.num_paths
        if n < 2:
            raise IndexError('Need at least 2 samples to form source-target pairs.')
        max_index = n * (n - 1)
        if index < 0 or index >= max_index:
            raise IndexError(f'Pair index out of range: {index}')
        src_idx = index // (n - 1)
        tgt_idx = index % (n - 1)
        if tgt_idx >= src_idx:
            tgt_idx += 1
        return self.paths[src_idx], self.paths[tgt_idx]
    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        # x_index = index // (len(self.paths) - 1)
        # s = index % (len(self.paths) - 1)
        # y_index = s + 1 if s >= x_index else s
        path_x, path_y = self._pair_from_index(index)
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)

        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return self.num_paths * (self.num_paths - 1)


class LPBABrainInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms
        self.num_paths = len(self.paths)

    def _pair_from_index(self, index):
        n = self.num_paths
        if n < 2:
            raise IndexError('Need at least 2 samples to form source-target pairs.')
        max_index = n * (n - 1)
        if index < 0 or index >= max_index:
            raise IndexError(f'Pair index out of range: {index}')
        src_idx = index // (n - 1)
        tgt_idx = index % (n - 1)
        if tgt_idx >= src_idx:
            tgt_idx += 1
        return self.paths[src_idx], self.paths[tgt_idx]
    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path_x, path_y = self._pair_from_index(index)
        # print(os.path.basename(path_x), os.path.basename(path_y))
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return self.num_paths * (self.num_paths - 1)
