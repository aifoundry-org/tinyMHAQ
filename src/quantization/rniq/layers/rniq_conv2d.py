import math
from tinygrad import nn, Tensor
from src.quantization.qtensor import QTensor
from src.quantization.rniq.rniq import Quantizer


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
        log_s_init: float = -12,
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

        self.log_wght_s = Tensor([log_s_init], requires_grad=True)
        # self.log_wght_s = Tensor([log_s_init], requires_grad=False)

        self.Q = Quantizer(self.log_wght_s.exp2(), Tensor(0),
                           Tensor(math.inf), Tensor(-math.inf))

    def __call__(self, x):
        s = self.log_wght_s.exp2()
        min = self.weight.min()

        self.Q.scale = s
        self.Q.zero_point = min

        weight = self.Q.quantize(self.Q.dequantize(self.weight))

        return x.conv2d(weight,
                        self.bias, self.groups, self.stride, self.dilation, self.padding)

