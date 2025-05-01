import sys
import os
import numpy as np
import time

from typing import Tuple
from tinygrad.helpers import tqdm, trange
from tinygrad import nn, Tensor, TinyJit, dtypes, GlobalCounters, Device
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.data.datasets.cifar10 import Cifar10Dataset
from src.models.resnet.speedconvnet import SpeedyConvNet
from tinygrad.helpers import Context
from src.utils.gpu_mem import measure_gpu_mem


if __name__ == "__main__":

    dataset = Cifar10Dataset(data_dir=os.path.dirname(os.path.dirname(os.path.realpath(__file__))), batch_size=64, val_split=0.0)
    train_dataloader = dataset.get_train_dataloader()

    model = SpeedyConvNet()
    opt = optim.Adam(get_parameters(model), lr=1e-6)
    loss_fn = lambda out,y: out.cross_entropy(y)
    
    @measure_gpu_mem()
    def vanilla_train():

        batchsize = 64
        def vanilla_dataset():
            X_train, Y_train, _, _ = nn.datasets.cifar()
            num_steps_per_epoch      = X_train.size(0) // batchsize
            idxs = np.arange(X_train.shape[0])
            tidxs = Tensor(idxs, dtype='int')[:num_steps_per_epoch*batchsize].reshape(num_steps_per_epoch, batchsize)
            # TODO: without this line indexing doesn't fuse!
            # X_train, Y_train, X_test, Y_test = [x.contiguous() for x in [X_train, Y_train, X_test, Y_test]]
            # X_train, Y_train = [x.contiguous() for x in [X_train, Y_train]]
            # for i in trange(len(tidxs)):
                # X_train[tidxs[i]], Y_train[tidxs[i]]
            
            return X_train, Y_train, tidxs

        X_train, Y_train, tidx = vanilla_dataset()
        cifar10_std, cifar10_mean = X_train.float().std_mean(axis=(0, 2, 3))
        def preprocess(X:Tensor, Y:Tensor) -> Tuple[Tensor, Tensor]:
            return ((X - cifar10_mean.view(1, -1, 1, 1)) / cifar10_std.view(1, -1, 1, 1)).cast(dtypes.default_float), Y.one_hot(10)

        @TinyJit
        @Tensor.train()
        def new_train() -> Tensor:
            X, Y = next(train_dataloader)
            out = model(X)
            loss = loss_fn(out, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            return loss / (batchsize)

        @TinyJit
        @Tensor.train()
        def train_step(idxs:Tensor) -> Tensor:
            with Context(SPLIT_REDUCEOP=0, FUSE_ARANGE=1):
                X = X_train[idxs]
                Y = Y_train[idxs].realize(X)
            X, Y = preprocess(X, Y)
            out = model(X)
            loss = loss_fn(out, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            return loss / (batchsize)
        
        for epoch in range(1):
            for step in range(20):
                gst = time.perf_counter()
                # loss = train_step(tidx[step].contiguous())
                loss = train_step(tidx[step])
                # loss = new_train()
                GlobalCounters.reset()
                print(time.perf_counter() - gst)
    
    vanilla_train()