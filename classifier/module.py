import pytorch_lightning as pl
from torch.utils.data import DataLoader

__all__ = ['ClassifierLightning']


class ClassifierLightning(pl.LightningModule):
    def __init__(self, batch_size, num_workers,
                 optimizer_cls, scheduler_cls,
                 train_ds, test_ds,
                 model, loss,
                 num_classes):
        super().__init__()

        self.batch_size = batch_size
        self.save_hyperparameters('batch_size')

        self.num_workers = num_workers
        self.save_hyperparameters('num_workers')

        self.model = model

        self.loss_name, self.loss = loss
        self.save_hyperparameters('loss')

        self.optimizer_cls = optimizer_cls
        self.save_hyperparameters('optimizer_cls')
        self.scheduler_cls = scheduler_cls
        self.save_hyperparameters('scheduler_cls')

        self.train_ds = train_ds
        self.test_ds = test_ds

        self.num_classes = num_classes
        self.save_hyperparameters('num_classes')
        self.num_workers = num_workers
        self.save_hyperparameters('num_workers')

        self.metrics = [
            pl.metrics.classification.accuracy,
            pl.metrics.functional.f1
        ]

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, pin_memory=True,
                          num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y_true = batch['image'], batch['label']
        y_pred = self(x)

        loss = self.loss(y_pred, y_true)
        res_kwargs = dict(prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log(self.loss_name, loss, **res_kwargs)

        # y_pred_confs = y_pred.sigmoid()
        # for metric in self.metrics:
        #     self.log(f'train/{type(metric).__name__.lower()}',
        #              metric(y_pred_confs, y_true), **res_kwargs)
        return loss

    def test_step(self, batch, batch_nb):
        x, y_true = batch['image'], batch['label']
        y_pred = self(x)

        # res_kwargs = dict(prog_bar=True, on_step=False, on_epoch=True, logger=True)
        # for metric in self.metrics:
        #     self.log(f'test/{type(metric).__name__.lower()}',
        #              metric(y_pred, y_true), **res_kwargs)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(params=self.parameters())
        self.scheduler_cls['scheduler'] = self.scheduler_cls['scheduler'](optimizer=optimizer)
        return [optimizer], [self.scheduler_cls]
