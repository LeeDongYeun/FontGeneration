"""
DMFont
Copyright (c) 2020-present NAVER Corp.
"""
import torch.nn as nn


def split_dim(x, dim, n_chunks):
    shape = x.shape
    assert shape[dim] % n_chunks == 0
    return x.view(*shape[:dim], n_chunks, shape[dim] // n_chunks, *shape[dim+1:])


def weights_init(init_type='default'):
    """Adopted and modified from FUNIT

    Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
    Licensed under the CC BY-NC-SA 4.0 license
    (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
    """
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=2**0.5)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=2**0.5)
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    return init_fun


def spectral_norm(module):
    """ init & apply spectral norm """
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    return nn.utils.spectral_norm(module)
