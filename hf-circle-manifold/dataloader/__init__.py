
import math
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from custom_pkg.pytorch.operations import DataCycle


class Circle(object):
    """ The circle 1D manifold (curve) in 2D space."""
    def __init__(self, n_samples, angle1=math.pi/6.0, angle2=math.pi*5.0/6.0):
        assert 0 <= angle1 < angle2 <= math.pi
        # Config.
        self._angle_radius, self._angle_center = (angle2 - angle1) / 2.0, (angle1 + angle2) / 2.0
        angles = (np.random.rand(n_samples).astype("float32") * 2.0 - 1.0) * self._angle_radius + self._angle_center
        # Set data
        self._data = np.concatenate([np.cos(angles)[:, None], np.sin(angles)[:, None]], axis=1)[:, :, None, None]

    def sampling_u(self, batch_size, device):
        u = torch.rand(batch_size, 1, 1, 1, device=device)*2.0 - 1.0
        u = u * self._angle_radius
        # Return
        return u

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


def generate_data(cfg):
    # 1. Get dataset.
    if cfg.args.dataset == 'circle':
        dataset = Circle(n_samples=cfg.args.dataset_n_samples)
    else:
        raise NotImplementedError
    # 2. Get dataloader.
    dataloader = DataLoader(
        dataset, batch_size=cfg.args.batch_size, drop_last=cfg.args.dataset_drop_last,
        shuffle=cfg.args.dataset_shuffle, num_workers=cfg.args.dataset_num_threads)
    # Return
    return DataCycle(dataloader)
