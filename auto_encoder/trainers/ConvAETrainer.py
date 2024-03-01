"""
@author: bochengz
@date: 2024/02/26
@email: bochengzeng@bochengz.top
"""
from common.trainers.Trainer import Trainer
import torch
import torch.nn as nn
from kogger import Logger

logger = Logger.get_logger(__name__)


class ConvAETrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, config,
                 loss_func=nn.MSELoss()):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            loss_func=loss_func
        )

    def train(self, tr_loader, val_loader):
        self.model.train()
        min_val_loss = 1.0e+6
        for epoch in range(self.config.start_epoch,
                           self.config.epochs + self.config.start_epoch):
            tr_loss = 0
            for b_idx, (x, visc) in enumerate(tr_loader):
                x = x.to(self.config.device)
                visc = visc.to(self.config.device)
                _, recovery = self.model(x, visc)
                loss = self.loss_func(x, recovery)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    tr_loss += loss.item()

            self.scheduler.step()

            if epoch % self.config.print_freq == 0:
                logger.info('[Epoch {:4d}/{}] tr_loss: {:.2e}'
                            .format(epoch, self.config.epochs +
                                    self.config.start_epoch - 1,
                                    tr_loss / len(tr_loader)))

            if epoch % self.config.save_freq == 0:
                self.save_checkpoint()

            if epoch == self.config.start_epoch or \
                    epoch % self.config.eval_freq == 0:
                loss = self.evaluate(val_loader)
                info_str = '[Epoch {:4d}/{}] val_loss: {:.2e}'\
                    .format(epoch, self.config.epochs +
                            self.config.start_epoch - 1, loss)
                if loss < min_val_loss:
                    min_val_loss = loss
                    self.save_checkpoint(val=True)
                    info_str = info_str + ' [MIN]'
                logger.info(info_str)

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0
        for b_idx, (x, visc) in enumerate(val_loader):
            x = x.to(self.config.device)
            visc = visc.to(self.config.device)
            _, recovery = self.model(x, visc)
            loss = self.loss_func(x, recovery)
            val_loss += loss.item()
        return val_loss / len(val_loader)
