import math
from tinygrad import nn, Tensor
from src.quantization.rniq.rniq import Quantizer

class NoisyAct:
    def __init__(self, 
                 init_s=-10,
                 init_q=10,
                 signed=True,
                 disable=False):
        self.disable = disable
        self.signed = signed
        
        self._act_b = Tensor([0]).float()
        self._log_act_s = Tensor([init_s]).float()
        self._log_act_q = Tensor([init_q]).float()
        
        if self.signed:
            self.act_b = Tensor(self._act_b, requires_grad=True)
        else:
            self.act_b = Tensor(self._act_b, requires_grad=False)
        

        self.log_act_q = Tensor(self._log_act_q, requires_grad=True)
        self.log_act_s = Tensor(self._log_act_s, requires_grad=True)

        self.Q = Quantizer(self._log_act_s.exp2(), Tensor.zeros(1), Tensor(-math.inf), Tensor(math.inf))

    def __call__(self, x):
        if self.disable:
            return x
        
        s = self.log_act_s.exp2()
        q = self.log_act_q.exp2()

        self.Q.zero_point = self.act_b
        self.Q.min_val = self.act_b
        self.Q.max_val = self.act_b + q - s


        return self.Q.dequantize(self.Q.quantize(x))