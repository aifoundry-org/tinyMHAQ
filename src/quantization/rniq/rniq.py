from tinygrad.tensor import Tensor


class Quantizer:
    def __init__(self, scale: Tensor, zero_point: Tensor, min_val: Tensor, max_val: Tensor):
        self.scale = Tensor(scale.item(), requires_grad=False)
        self.zero_point = Tensor(zero_point.item(), requires_grad=False)
        self.min_val = Tensor(min_val.item(), requires_grad=False)
        self.max_val = Tensor(max_val.item(), requires_grad=False)

    def quantize(self, value: Tensor):
        value = value.clamp(min_=self.min_val, max_=self.max_val)

        value = (value - self.zero_point) / self.scale

        noise = (value - value.round()).nround()

        value = value + noise

        return value

    def dequantize(self, quantized_value):
        return quantized_value * self.scale + self.zero_point
