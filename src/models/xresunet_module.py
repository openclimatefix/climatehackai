from typing import Any, List

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from pytorch_lightning import LightningModule

from src.models.components.xresnet_unet import XResUNet


class XResUNetLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        #self.save_hyperparameters(logger=False)

        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay

        # loss function
        self.criterion = torch.nn.L1Loss()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        x = x.float()
        y = y.float()
        logits = self.forward(x)
        # Center crop here
        loss = self.criterion(logits[:, :, 16:48, 16:48], y[:, :, 16:48, 16:48])
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, }

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return AdamW(self.net.parameters(), lr=self.lr)
