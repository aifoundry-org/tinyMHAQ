import sys
import os

from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.models.resnet.resnet20 import resnet20_cifar10
from src.data.datasets.cifar10 import Cifar10Dataset
from src.training.trainer import TinyTrainer

if __name__ == "__main__":
    dataset = Cifar10Dataset("./", batch_size=64, val_split=0)
    model = resnet20_cifar10()
    trainer = TinyTrainer()
    trainer.epochs = 10
    trainer.optim = optim.Adam(get_parameters(model))
    trainer.loss_f = lambda out,y: out.sparse_categorical_crossentropy(y)

    trainer.fit(model, dataset)
    pass