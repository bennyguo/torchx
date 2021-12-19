import argparse
import os
import time
import yaml
import shutil

import datasets
import models

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback
from utils.loggers import ConsoleLogger
from utils.misc import prompt, parse_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--version', default=None)
    parser.add_argument('--gpu', default='0')
    # todo: distinguish between resume and load weights
    parser.add_argument('--resume')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = parse_config(args.config)

    save_name = args.name or '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    if os.path.exists(save_path):
        ans = prompt(f"Experiment directory {save_path} already exists. Override checkpoints?")
        if not ans:
            exit(1)

    config.cmd_args = vars(args)

    if 'seed' not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed)
    n_gpus = len(args.gpu.split(','))

    dm = datasets.make(config.dataset.name, config.dataset)
    system = models.make(config.system.name, config.system)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_path, 'ckpt'),
            save_top_k=-1, # save all checkpoints,
            every_n_epochs=1
        ),
        CodeSnapshotCallback(
            os.path.join(save_path, 'code')
        ),
        ConfigSnapshotCallback(
            os.path.join(save_path, 'config'), config
        )
    ]

    loggers = [
        TensorBoardLogger(save_path, name='runs'),
        ConsoleLogger(log_keys=['val'])
    ]

    trainer = Trainer(
        gpus=n_gpus,
        callbacks=callbacks,
        logger=loggers,
        strategy=DDPPlugin(find_unused_parameters=False) if config.strategy == 'ddp' else config.strategy,
        **config.trainer
    )

    if args.resume:
        trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
    
    trainer.fit(system, datamodule=dm)


if __name__ == '__main__':
    main()
