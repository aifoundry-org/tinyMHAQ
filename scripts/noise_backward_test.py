import tinygrad
from tinygrad import Tensor, nn, Device


if __name__ == "__main__":
    # Device.DEFAULT = "CPU"
    for i in range(10):
        with Tensor.train():
            x = Tensor([2.9], requires_grad=True)
            y = x.nround().round()
            y.backward(gradient=Tensor([0.0]))

        print(f"y = {y.numpy()}, x grad = {x.grad.tolist()}")