"""
@author: bochengz
@date: 2024/02/26
@email: bochengzeng@bochengz.top
"""
import os
import os.path as osp
from abc import abstractmethod
import torch
from kogger import Logger

logger = Logger.get_logger(__name__)


class Trainer:
    """
    Abstract class for train

    Args:
        model (torch.nn.Module): the instance of pytorch module's child class
        optimizer (torch.optim.Optimizer): pytorch optimizer
        scheduler (torch.optim.lr_scheduler.LRScheduler): pytorch scheduler
        config (Namespace): the config of project
    """

    def __init__(self, model, optimizer, scheduler, config, loss_func):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.loss_func = loss_func

    @abstractmethod
    def train(self, tr_loader, val_loader):
        raise NotImplementedError('train function has not been implemented')

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError('evaluate function has not been implemented')

    def save_checkpoint(self, val=False):
        if val:
            ckpt_path = self.config.val_ckpt_path
        else:
            ckpt_path = self.config.tr_ckpt_path

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, ckpt_path)

    def load_checkpoint(self, val=False):
        if val:
            ckpt_path = self.config.val_ckpt_path
        else:
            ckpt_path = self.config.tr_ckpt_path

        if osp.isfile(ckpt_path):
            logger.info('Load model from checkpoint: {}'
                        .format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            raise FileNotFoundError('Provided path of checkpoint: {} does not'
                                    'exist'.format(ckpt_path))
