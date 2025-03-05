from tinygrad.helpers import partition, trange, getenv, Context
from tinygrad import Tensor, nn, GlobalCounters, TinyJit, dtypes
import numpy as np
from typing import Tuple, cast
import math
import time
start_tm = time.perf_counter()
# from extra.lr_scheduler import OneCycleLR

dtypes.default_float = dtypes.half

# from https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
batchsize = getenv("BS", 1024)
bias_scaler = 64
hyp = {
    'opt': {
        # TODO: Is there maybe a better way to express the bias and batchnorm scaling? :'))))
        'bias_lr':        1.525 * bias_scaler/512,
        'non_bias_lr':    1.525 / 512,
        'bias_decay':     6.687e-4 * batchsize/bias_scaler,
        'non_bias_decay': 6.687e-4 * batchsize,
        'scaling_factor': 1./9,
        'percent_start': .23,
        # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
        'loss_scale_scaler': 1./32,
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        'batch_norm_momentum': .4,
        'cutmix_size': 3,
        'cutmix_epochs': 6,
        'pad_amount': 2,
        'base_depth': 64  # This should be a factor of 8 in some way to stay tensor core friendly
    },
    'misc': {
        'ema': {
            'epochs': 10,  # Slight bug in that this counts only full epochs and then additionally runs the EMA for any fractional epochs at the end too
            'decay_base': .95,
            'decay_pow': 3.,
            'every_n_steps': 5,
        },
        'train_epochs': 12,
        # 'train_epochs': 12.1,
        'device': 'cuda',
        'data_location': 'data.pt',
    }
}

# You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
scaler = 2.
depths = {
    # 32  w/ scaler at base value
    'init':   round(scaler**-1*hyp['net']['base_depth']),
    # 64  w/ scaler at base value
    'block1': round(scaler ** 0*hyp['net']['base_depth']),
    # 256 w/ scaler at base value
    'block2': round(scaler ** 2*hyp['net']['base_depth']),
    # 512 w/ scaler at base value
    'block3': round(scaler ** 3*hyp['net']['base_depth']),
    'num_classes': 10
}
whiten_conv_depth = 3*hyp['net']['whitening']['kernel_size']**2


class ConvGroup:
    def __init__(self, channels_in, channels_out):
        self.conv1 = nn.Conv2d(channels_in, channels_out,
                               kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels_out, channels_out,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm(channels_out, track_running_stats=False,
                                  eps=1e-12, momentum=hyp['net']['batch_norm_momentum'])
        self.norm2 = nn.BatchNorm(channels_out, track_running_stats=False,
                                  eps=1e-12, momentum=hyp['net']['batch_norm_momentum'])
        cast(Tensor, self.norm1.weight).requires_grad = False
        cast(Tensor, self.norm2.weight).requires_grad = False

    def __call__(self, x: Tensor) -> Tensor:
        x = self.norm1(self.conv1(x).max_pool2d().float()).cast(
            dtypes.default_float).quick_gelu()
        return self.norm2(self.conv2(x).float()).cast(dtypes.default_float).quick_gelu()


class SpeedyConvNet:
    def __init__(self):
        self.whiten = nn.Conv2d(
            3, 2*whiten_conv_depth, kernel_size=hyp['net']['whitening']['kernel_size'], padding=0, bias=False)
        self.conv_group_1 = ConvGroup(2*whiten_conv_depth, depths['block1'])
        self.conv_group_2 = ConvGroup(depths['block1'], depths['block2'])
        self.conv_group_3 = ConvGroup(depths['block2'], depths['block3'])
        self.linear = nn.Linear(
            depths['block3'], depths['num_classes'], bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.whiten(x).quick_gelu()
        x = x.sequential(
            [self.conv_group_1, self.conv_group_2, self.conv_group_3])
        return self.linear(x.max(axis=(2, 3))) * hyp['opt']['scaling_factor']

