"""
@author: bochengz
@date: 2024/02/26
@email: bochengzeng@bochengz.top
"""
import torch.nn as nn
from abc import abstractmethod


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

    @abstractmethod
    def embed(self, *args, **kwargs):
        raise NotImplementedError('embed function has not been properly '
                                  'override')

    @abstractmethod
    def recover(self, *args, **kwargs):
        raise NotImplementedError('recover function has not been properly '
                                  'override')

    @property
    def num_parameters(self):
        return sum([param.nelement() for param in self.parameters()])