import os
import time
import gc
import sys
import numpy as np

from tinygrad import Tensor, dtypes, TinyJit, nn
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import Context

sys.path.append(os.getcwd())
from src.utils.gpu_mem import measure_gpu_mem
from src.data.datasets.cifar10 import Cifar10Dataset
from src.models.resnet.speedconvnet import SpeedyConvNet

BATCH_SIZE = 64
STEPS      = 10
LR         = 1e-6

def preprocess(X, Y, mean, std):
    X = ((X - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)).cast(dtypes.default_float)
    return X, Y.one_hot(10)

@measure_gpu_mem()
def explicit_pipeline():
    model   = SpeedyConvNet()
    opt     = optim.Adam(get_parameters(model), lr=LR)
    loss_fn = lambda out, y: out.cross_entropy(y)

    X_train, Y_train, _, _ = nn.datasets.cifar()
    total = (X_train.shape[0] // BATCH_SIZE) * BATCH_SIZE
    idxs  = Tensor(np.arange(total).reshape(-1, BATCH_SIZE), dtype=dtypes.int32)
    std, mean = X_train.float().std_mean(axis=(0, 2, 3))

    @TinyJit
    @Tensor.train()
    def train_step(idx: Tensor):
        with Context(SPLIT_REDUCEOP=0, FUSE_ARANGE=1):
            Xb = X_train[idx].contiguous()
            Yb = Y_train[idx].contiguous().realize(Xb)
        Xb, Yb = preprocess(Xb, Yb, mean, std)
        out = model(Xb)
        loss = loss_fn(out, Yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss

    for i in range(STEPS):
        t0 = time.perf_counter()
        train_step(idxs[i].contiguous())
        print(f"[Explicit] Step {i+1}/{STEPS}: {(time.perf_counter() - t0)*1e3:.2f} ms")

    del model, opt, X_train, Y_train, idxs, std, mean, train_step
    gc.collect()

@measure_gpu_mem()
def cifar10_dataset_pipeline():
    model   = SpeedyConvNet()
    opt     = optim.Adam(get_parameters(model), lr=LR)
    loss_fn = lambda out, y: out.cross_entropy(y)

    ds = Cifar10Dataset(data_dir=os.getcwd(), batch_size=BATCH_SIZE, val_split=0.0)
    dl = ds.get_train_dataloader()

    @TinyJit
    @Tensor.train()
    def train_step(Xb: Tensor, Yb: Tensor):
        out = model(Xb)
        loss = loss_fn(out, Yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss

    for i in range(STEPS):
        Xb, Yb = next(dl)
        t0 = time.perf_counter()
        train_step(Xb, Yb)
        print(f"[Dataset ] Step {i+1}/{STEPS}: {(time.perf_counter() - t0)*1e3:.2f} ms")

    del model, opt, ds, dl, train_step
    gc.collect()

if __name__ == "__main__":
    from tinygrad.device import Device
    print(f"Measuring pipeline with batch_size={BATCH_SIZE} and steps={STEPS}")
    # Device.DEFAULT = "LLVM"
    print("\n=== Cifar10Dataset Pipeline ===")
    cifar10_dataset_pipeline()

    # print("=== Explicit CIFAR Pipeline ===")
    # explicit_pipeline()
    

