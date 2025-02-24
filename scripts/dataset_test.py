import sys
import os

from tinygrad.helpers import tqdm, trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.data.datasets.cifar10 import Cifar10Dataset

if __name__ == "__main__":
    dataset = Cifar10Dataset(data_dir=os.path.dirname(os.path.dirname(os.path.realpath(__file__))), batch_size=64)
    # print(next(dataset.get_test_batch()))
    for i in trange(dataset.get_test_dataloader()):
        pass
    # print(next(dataset.get_train_batch()))