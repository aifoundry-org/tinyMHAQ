import sys
import os

from tinygrad import tensor

from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.models.resnet.resnet20 import resnet20_cifar10
from src.models.resnet.speedconvnet import SpeedyConvNet
from src.data.datasets.cifar10 import Cifar10Dataset
from src.data.metrics.accuracy import Accuracy
from src.training.trainer import TinyTrainer

if __name__ == "__main__":
    dataset = Cifar10Dataset("./", batch_size=128, val_split=0, device="AMD")
    # model = resnet20_cifar10()
    model = SpeedyConvNet()
    model.to("AMD")
    trainer = TinyTrainer()
    trainer.device = "AMD"
    trainer.train_metrics = [Accuracy()]
    # trainer.epochs = 2
    trainer.steps = 10
    # trainer.optim = optim.Adam(get_parameters(model))
    trainer.optim = optim.SGD(get_parameters(model))
    trainer.loss_f = lambda out,y: out.cross_entropy(y)

    trainer.fit(model, dataset)