"""
@author: bochengz
@date: 2024/02/26
@email: bochengzeng@bochengz.top
"""
from abc import abstractmethod
from kogger import Logger


class DataHandler:
    """
    Abstract class for data handler, used to create the train, validate and \
    test datasets
    """
    def __init__(self):
        self.logger = Logger.get_logger('DataHandler')

    @abstractmethod
    def create_tr_loader(self, *args, **kwargs):
        raise NotImplementedError('create_tr_loader function has not been '
                                  'implemented')

    @abstractmethod
    def create_val_loader(self, *args, **kwargs):
        raise NotImplementedError('create_val_loader function has not been '
                                  'implemented')

    @abstractmethod
    def create_te_loader(self, *args, **kwargs):
        raise NotImplementedError('create_te_loader function has not been '
                                  'implemented')