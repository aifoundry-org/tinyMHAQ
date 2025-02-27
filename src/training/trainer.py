import time
import numpy as np

from src.types.dataset import Dataset
from typing import Tuple
from tinygrad.tensor import Tensor
from tinygrad.device import ALL_DEVICES
from tinygrad.helpers import trange, getenv


class TinyTrainer:
    def __init__(self):
        self.batch_size = 64
        self.epochs = getenv("EPOCHS", 2048)
        self.loss_f = None
        self.optim = None
        self.train_metrics = []
        self.val_metrics = []
        self.device: str = "LLVM"

        if self.device not in ALL_DEVICES:
            raise AssertionError(f"Wrong device '{self.device}', available options are: {ALL_DEVICES}")
        

    def fit(self, model, dataset: Dataset):
        train_dataloader = dataset.get_train_dataloader()

        with Tensor.train():
            for i in (t:= trange(self.epochs)):
                for _ in (bt:= trange(len(train_dataloader))):
                    input, target = next(train_dataloader)

                    out = model(input.to(self.device))

                    loss = self.loss_f(out, target.to(self.device))

                    self.optim.zero_grad()

                    loss.backward()

                    self.optim.step()

                    bt.set_description(f"Loss = {loss.numpy():.5f}")

                    for metric in self.train_metrics:
                        metric_value = metric(out, target)
                        bt.set_description(bt.desc + f"{metric.__class__.__name__} = {metric_value}")
                
                train_dataloader.reset()

               
                



    def validate(self, model, dataset: Dataset):

        val_dataloader = dataset.get_val_dataloader()

        Tensor.training = False

        for _ in (bt:= trange(len(val_dataloader))):
            input, target = next(val_dataloader)

            out = model(input.to(self.device))

            loss = self.loss_f(out, target.to(self.device))

            bt.set_description(f"Val Loss = {loss.numpy():.5f}")

            for metric in self.val_metrics:
                metric_value = metric(out, target)               
                bt.set_description(bt.desc + f"{metric.__class__.__name__} = {metric_value}")

        val_dataloader.reset()


    def test(self, model, dataset: Dataset):

        test_dataloader = dataset.get_test_dataloader()

        Tensor.training = False

        for _ in (bt:= trange(len(test_dataloader))):
            input, target = next(test_dataloader)

            out = model(input.to(self.device))

            loss = self.loss_f(out, target.to(self.device))

            bt.set_description(f"Test loss = {loss.numpy():.5f}")

            for metric in self.val_metrics:
                metric_value = metric(out, target)
                bt.set_description(bt.desc + f"{metric.__class__.__name__} = {metric_value}")
        
        test_dataloader.reset()
