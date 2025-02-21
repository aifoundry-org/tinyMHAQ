import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.data.datasets.cifar10 import Cifar10Dataset

if __name__ == "__main__":
    dataset = Cifar10Dataset(data_dir=os.path.dirname(os.path.dirname(os.path.realpath(__file__))), batch_size=64)
    print(next(dataset.get_test_batch()))
    print(next(dataset.get_train_batch()))