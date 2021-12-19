import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import models
from .criterions import ClassificationAccuracy

import pytorch_lightning as pl


@models.register('mnist')
class MNISTSystem(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = models.make(config.model.name, config)
        self.criterions = {
            'ce': nn.CrossEntropyLoss(),
            'acc': ClassificationAccuracy()
        }
    
    def forward(self, batch):
        return self.model(batch['img'])
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.criterions['ce'](out, batch['label']) * self.config.loss.lambda_ce
        # log only executes on rank 0
        self.log('train/loss', loss, on_step=True, prog_bar=True)
        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.criterions['ce'](out, batch['label']) * self.config.loss.lambda_ce
        acc = self.criterions['acc'](out, batch['label'])
        return {
            'loss': loss,
            'acc': acc,
            'batch_size': batch['label'].size(0)
        }
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            count = sum(o['batch_size'].sum().item() for o in out)
            loss = sum((o['loss'] * o['batch_size']).sum().item() for o in out) / count
            acc = sum((o['acc'] * o['batch_size']).sum().item() for o in out) / count
            self.log('val/loss', loss, on_epoch=True, prog_bar=True, rank_zero_only=True)
            self.log('val/acc', acc, on_epoch=True, prog_bar=True, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        pass
    
    def test_epoch_end(self, out):
        pass

    def configure_optimizers(self):
        print('configure optimizers')
        optim = getattr(torch.optim, self.config.optimizer.name)(self.model.parameters(), **self.config.optimizer.args)
        ret = {
            'optimizer': optim,
        }

        if 'scheduler' in self.config:
            sched = getattr(torch.optim.lr_scheduler, self.config.scheduler.name)(optim, **self.config.scheduler.args)
            ret.update({
                'lr_scheduler': sched,
            })

        return ret
