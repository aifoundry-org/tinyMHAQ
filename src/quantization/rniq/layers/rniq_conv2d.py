from tinygrad import nn
from src.quantization.qtensor import QTensor


class NoisyConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        weight = QTensor(
            data=self.weight.data,
            device=self.weight.device,
            dtype=self.weight.dtype,
            requires_grad=self.weight.requires_grad,
        )
        self.weight = weight
    
    def __call__(self, x):
        return super().__call__(x)
