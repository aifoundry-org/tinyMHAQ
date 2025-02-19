import os
import numpy as np

from abc import abstractmethod
from tinygrad.tensor import Tensor
from tinygrad.helpers import tqdm


class ImageClassificationDataset:
    def __init__(
        self,
        data_dir,
        batch_size,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.dataset_path = os.path.join(self.data_dir, self.__class__.__name__)

        self._train_passes = 0
        self._val_passes = 0
        self._test_passes = 0

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_train_batch(self):
        pass

    @abstractmethod
    def get_val_batch(self):
        pass

    @abstractmethod
    def get_test_batch(self):
        pass

    @abstractmethod
    def get_train_transforms(self, input: Tensor) -> Tensor:
        pass

    def _get_sample_seq(self, size: int, random: bool = False) -> np.ndarray:
        if random:
            return np.random.choice(size, size, replace=False)
        else:
            return np.arange(size)

    def _load_disk_tensor(self, db_list, val_split=0.2, seed=None, postfix=""):
        total_samples = sum(db[b"data"].shape[0] for db in db_list)

        first_x = db_list[0][b"data"]
        first_y = np.array(db_list[0][b"labels"])

        # Determine the overall shape for X and Y.
        X_shape = (total_samples,) + first_x.shape[1:]
        Y_shape = (
            (total_samples,)
            if first_y.ndim == 1
            else (total_samples,) + first_y.shape[1:]
        )

        X_np = np.empty(X_shape, dtype=first_x.dtype)
        Y_np = np.empty(Y_shape, dtype=first_y.dtype)

        idx = 0
        for db in db_list:
            x = db[b"data"]
            y = np.array(db[b"labels"])
            assert x.shape[0] == y.shape[0], "Mismatch between data and labels"
            X_np[idx : idx + x.shape[0]] = x
            Y_np[idx : idx + x.shape[0]] = y
            idx += x.shape[0]
        assert idx == total_samples, "Not all samples were loaded"

        indices = np.arange(total_samples)
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

        if val_split == 0:
            X = Tensor(
                X_np[indices], device=f"disk:{self.dataset_path}/input_{postfix}"
            )
            Y = Tensor(
                Y_np[indices], device=f"disk:{self.dataset_path}/target_{postfix}"
            )
            return X, Y

        num_val = int(total_samples * val_split)
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        X_train = Tensor(
            X_np[train_indices], device=f"disk:{self.dataset_path}/input_train"
        )
        Y_train = Tensor(
            Y_np[train_indices], device=f"disk:{self.dataset_path}/target_train"
        )
        X_val = Tensor(X_np[val_indices], device=f"disk:{self.dataset_path}/input_val")
        Y_val = Tensor(Y_np[val_indices], device=f"disk:{self.dataset_path}/target_val")

        return X_train, Y_train, X_val, Y_val
