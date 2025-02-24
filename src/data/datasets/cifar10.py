import os, gzip, tarfile, pickle
import numpy as np

from typing import Generator

from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch, tqdm
from src.types.image_classification_dataset import ImageClassificationDataset
from src.types.dataloader import Dataloader
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

        self.train_dataloader = Dataloader(self.X_train, self.Y_train, self.batch_size, shuffle=True, transforms=self.transform_train)
        self.val_dataloader = Dataloader(self.X_val, self.Y_val, self.batch_size, shuffle=False, transforms=self.transform_val)
        self.test_dataloader = Dataloader(self.X_test, self.Y_test, self.batch_size, shuffle=False, transforms=self.transform_val)
        return super().setup()
    
    # The idea with generator is pretty straightforward
    # 1. Extracted data lies in RAM
    # 2. Subset of that data is transformed with target transformations/augmentations
    # then moved to the target device.
    #
    # Good luck processing imagenet or something of a similar size, because 
    # Tensor slicing is not supported for tensors stored on disk.

    def get_train_dataloader(self) -> Generator:
        return self.train_dataloader

    def get_val_dataloader(self) -> Generator:
        return self.val_dataloader

    def dataloader(self) -> Generator:
        return self.test_dataloader