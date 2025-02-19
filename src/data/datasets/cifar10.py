import os, gzip, tarfile, pickle
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch
from src.types.data import ImageClassificationDataset
from src.data.transforms import (
    ComposeTransforms,
    image_random_horizontal_flip,
    image_random_crop,
    image_normalize,
)


class Cifar10Dataset(ImageClassificationDataset):
    def __init__(self, data_dir, batch_size, val_split=0.2):

        self.val_split = val_split
        self.cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
        self.cifar_std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
        self.transform_train = ComposeTransforms[
            image_random_horizontal_flip(), 
            image_random_crop(32, padding=4, padding_mode="reflect"), 
            image_normalize(mean=self.cifar_mean, std=self.cifar_std)
        ]

        super().__init__(data_dir, batch_size)

    def setup(self):
        fn = fetch("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        tt = tarfile.open(fn, mode="r:gz")
        if not os.path.isfile(f"{self.data_dir}/cifar_extracted"):
            self.X_test, self.Y_test = self._load_disk_tensor(
                [
                    pickle.load(
                        tt.extractfile("cifar-10-batches-py/test_batch"),
                        encoding="bytes",
                    )
                ],
                val_split=0,
                postfix="test",
            )

            if self.val_split:
                self.X_train, self.Y_train, self.X_val, self.Y_val = (
                    self._load_disk_tensor(
                        [
                            pickle.load(
                                tt.extractfile(f"cifar-10-batches-py/data_batch_{i}"),
                                encoding="bytes",
                            )
                            for i in range(1, 6)
                        ],
                        val_split=self.val_split,
                    )
                )
            else:
                self.X_train, self.Y_train = self._load_disk_tensor(
                    [
                        pickle.load(
                            tt.extractfile(f"cifar-10-batches-py/data_batch_{i}"),
                            encoding="bytes",
                        )
                        for i in range(1, 6)
                    ],
                    val_split=0,
                    postfix="train",
                )

                self.X_val, self.Y_val = self.X_test, self.Y_test

        return super().setup()

    def get_train_batch(self):
        return super().get_train_batch()

    def get_val_batch(self):
        return super().get_val_batch()

    def get_test_batch(self):
        return super().get_test_batch()


def fetch_cifar():
    X_train = Tensor.empty(
        50000, 3 * 32 * 32, device=f"disk:/tmp/cifar_train_x", dtype=dtypes.uint8
    )
    Y_train = Tensor.empty(50000, device=f"disk:/tmp/cifar_train_y", dtype=dtypes.int64)
    X_test = Tensor.empty(
        10000, 3 * 32 * 32, device=f"disk:/tmp/cifar_test_x", dtype=dtypes.uint8
    )
    Y_test = Tensor.empty(10000, device=f"disk:/tmp/cifar_test_y", dtype=dtypes.int64)

    if not os.path.isfile("/tmp/cifar_extracted"):

        def _load_disk_tensor(X, Y, db_list):
            idx = 0
            for db in db_list:
                x, y = db[b"data"], np.array(db[b"labels"])
                assert x.shape[0] == y.shape[0]
                X[idx : idx + x.shape[0]].assign(x)
                Y[idx : idx + x.shape[0]].assign(y)
                idx += x.shape[0]
            assert idx == X.shape[0] and X.shape[0] == Y.shape[0]

        print("downloading and extracting CIFAR...")
        fn = fetch("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        tt = tarfile.open(fn, mode="r:gz")
        _load_disk_tensor(
            X_train,
            Y_train,
            [
                pickle.load(
                    tt.extractfile(f"cifar-10-batches-py/data_batch_{i}"),
                    encoding="bytes",
                )
                for i in range(1, 6)
            ],
        )
        _load_disk_tensor(
            X_test,
            Y_test,
            [
                pickle.load(
                    tt.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes"
                )
            ],
        )
        open("/tmp/cifar_extracted", "wb").close()

    return X_train, Y_train, X_test, Y_test
