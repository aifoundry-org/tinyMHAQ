from tinygrad.tensor import Tensor

class QTensor(Tensor):
    def __init__(self, data, device = None, dtype = None, requires_grad = None):
        super().__init__(data, device, dtype, requires_grad)
    
    def backward(self, gradient = None):
        return super().backward(gradient)
    
    def quantize(self, scale: Tensor, zero_point: Tensor):
        pass

    def dequantize(self, scale: Tensor, zero_point: Tensor):
        pass