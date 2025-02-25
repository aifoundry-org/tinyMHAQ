import numpy as np

from typing import Tuple, Union
from tinygrad.tensor import Tensor


class Dataloader:
    def __init__(self, 
                 X: Tensor,
                 Y: Tensor, 
                 batch_size: int, 
                 shuffle: bool,
                 transforms: list, 
                 device: Union[str, None] = None):
        self.shuffle = shuffle
        self.idx = 0
        self.idcs = None
        self.ds_size = X.shape[0]
        self.batch_size = batch_size
        self.transforms = transforms
        self.X = X
        self.Y = Y
        self.device = device

        self.reset()

    def reset(self):
        self.X = self.X.to(None)
        self.Y = self.Y.to(None)

        if self.shuffle or not self.idcs:
            self.idcs = self._get_sample_idc()

        self.idx = 0

    def __next__(self):
        if self.idx >= self.ds_size:
            raise StopIteration
        
        batch_idx = self.idcs[self.idx : self.idx + self.batch_size]

        self.idx += self.batch_size
        
        return self.transforms(self.X[batch_idx]).to(self.device), self.Y[batch_idx].to(self.device)

    def __iter__(self):
        return self

    def __len__(self):
        return np.ceil(self.ds_size / self.batch_size).astype(int)

    def _get_sample_idc(self):
        if self.shuffle:
            return np.random.choice(self.ds_size, self.ds_size, replace=False).tolist()
        else:
            return np.arange(self.ds_size).tolist()
