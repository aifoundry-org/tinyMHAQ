import os, gzip, tarfile, pickle
import numpy as np

from typing import Generator

from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch, tqdm
from src.types.dataset import ImageClassificationDataset
from src.data.transforms import (
    ComposeTransforms,
    image_random_horizontal_flip,
    image_random_crop,
    image_reshape,
    image_normalize,
    to_tensor,
)


class Cifar10Dataset(ImageClassificationDataset):
    def __init__(self, data_dir, batch_size, val_split=0.2):

        self.val_split = val_split
        self.cifar_mean = Tensor(
            [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
        )
        self.cifar_std = Tensor(
            [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
        )
        self.transform_train = ComposeTransforms(
            [
                image_reshape((-1, 3, 32, 32)),
                # image_random_horizontal_flip(),
                # image_random_crop(32, padding=4, padding_mode="reflect"),
                image_normalize(mean=self.cifar_mean, std=self.cifar_std),
                # to_tensor()
            ]
        )

        self.transform_val = ComposeTransforms(
            [
                image_reshape((-1, 3, 32, 32)),
                image_normalize(mean=self.cifar_mean, std=self.cifar_std),
            ]
        )
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

        self.train_reset()
        self.val_reset()
        self.test_reset()
        return super().setup()

    def train_reset(self):
        self.X_train = self.X_train.to(None)
        self.Y_train = self.Y_train.to(None)
        self.train_samples = self._get_sample_seq(self.X_train.shape[0], random=True)

    def val_reset(self):
        self.X_val = self.X_val.to(None)
        self.Y_val = self.Y_val.to(None)
        self.val_samples = self._get_sample_seq(self.X_val.shape[0])

    def test_reset(self):
        self.X_test = self.X_test.to(None)
        self.Y_test = self.Y_test.to(None)
        self.test_samples = self._get_sample_seq(self.X_test.shape[0])
    
    # The idea with generator is pretty straightforward
    # 1. Extracted data lies in RAM
    # 2. Subset of that data is transformed with target transformations/augmentations
    # then moved to the target device.
    #
    # Good luck processing imagenet or something of a similar size, because 
    # Tensor slicing is not supported for tensors stored on disk.

    def get_train_batch(self) -> Generator:
        for i in range(0, len(self.train_samples), self.batch_size):
            yield (
                self.transform_train(
                    self.X_train[self.train_samples[i : i + self.batch_size]]
                ).to(self.device),
                self.Y_train[self.train_samples[i : i + self.batch_size]].to(
                    self.device
                ),
            )

    def get_val_batch(self) -> Generator:
        for i in range(0, len(self.val_samples), self.batch_size):
            yield (
                self.transform_val(
                    self.X_val[self.val_samples[i : i + self.batch_size]]
                ).to(self.device),
                self.Y_val[self.val_samples[i : i + self.batch_size]].to(self.device),
            )

    def get_test_batch(self) -> Generator:
        for i in range(0, len(self.test_samples), self.batch_size):
            yield (
                self.transform_val(
                    self.X_test[[self.test_samples[i : i + self.batch_size]]]
                ).to(self.device),
                self.Y_test[[self.test_samples[i : i + self.batch_size]]].to(
                    self.device
                ),
            )