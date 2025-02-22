import time
import numpy as np

from typing import Tuple
from tinygrad.tensor import Tensor
from tinygrad.helpers import trange, getenv

TINY = getenv("TINY")

class TinyTrainer:
    def __init__(self):
        self.batch_size = getenv("BS", 64 if TINY else 16)
        self.epochs = getenv("EPOCHS", 2048)
        self.loss_f = None
        self.optim = None
        self.train_metrics = []
        self.val_metrics = []
        

    def fit(self, model, dataset):
        # input, target = self._fetch_batch(data)
        
        with Tensor.train():
            for i in (t:= trange(self.epochs)):
                for batch in dataset.get_train_batch():
                    out = model(batch[0])

                    loss = self.loss_f(out, batch[1])

                    self.optim.zero_grad()

                    loss.backward()

                    self.optim.step()

                    print(loss.numpy())

                # for metric in self.train_metrics:
                #     metric_value = metric(out, target)
                #     # TODO metric logging
                
                # loss_ = loss.numpy()
                # TODO loss logging



    def validate(self, model, data):
        input, target = self._fetch_batch(data)

        Tensor.training = False
        
        out = model(input)

        val_loss = self.loss_f(out, target).numpy()

        for metric in self.val_metrics:
            metric_value = metric(out, target)

        # TODO logging validation loss and validation metrics


    def test(self, model, data):
        pass

    # def _fetch_batch(data) -> Tuple[Tensor, Tensor]:
        # pass