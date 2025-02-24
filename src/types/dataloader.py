import numpy as np

from typing import Tuple
from tinygrad.tensor import Tensor


class Dataloader:
    def __init__(self, XY: Tuple[Tensor, Tensor], 
                 batch_size: int, 
                 shuffle: bool):
        self.shuffle = shuffle
        self.idx = 0
        self.idcs = None
        self.ds_size = XY[0].shape[0]
        self.batch_size = batch_size

        self.reset()

    def reset(self):
        if self.shuffle or not self.idcs:
            self.idcs = self._get_sample_idc()

        self.idx = 0

    def __next__(self):
        if self.idx >= self.ds_size:
            raise StopIteration

    def __iter__(self):
        pass

    def __len__(self):
        return np.ceil(self.ds_size / self.batch_size)

    def _get_sample_idc(self):
        if self.shuffle:
            return np.random.choice(self.size, self.size, replace=False).tolist()
        else:
            return np.arange(self.size).tolist()
