from argparse import ArgumentParser
from pathlib import Path

import wandb
from pytorch_lightning import Trainer, loggers, callbacks

from mask_classification.classifier.module import ClassifierLightning
from mask_classification.classifier.utils import load_module

parser = ArgumentParser()
parser.add_argument('config', type=Path)
args = parser.parse_args()

if not args.config.exists():
    assert False, f"Config not found: {args.config}"

config = load_module(args.config)
config.base_kwargs['log_dir'].mkdir(parents=True, exist_ok=True)

logger = loggers.WandbLogger(save_dir=str(config.base_kwargs['log_dir']),
                             log_model=True, **config.wandb_kwargs)
logger.watch(config.lightning_module_kwargs['model'])
wandb.save(str(args.config))
trainer = Trainer(
    logger=logger,
    checkpoint_callback=True,
    default_root_dir=str(config.base_kwargs['log_dir']),
    callbacks=[callbacks.LearningRateMonitor(logging_interval='step')],
    **config.trainer_kwargs)
lightning_cls = ClassifierLightning
trainer.fit(lightning_cls(**config.lightning_module_kwargs))
