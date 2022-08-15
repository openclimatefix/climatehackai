from fastai.callback.tracker import *
from fastai.callback.wandb import *
from fastai.distributed import *
from fastai.layers import Mish
from fastai.vision.all import *
from fastai.vision.models.xresnet import *
from torch import nn


class XResUNet(nn.Module):
    def __init__(self, input_size, forecast_steps, history_steps, pretrained=False):
        super().__init__()
        arch = partial(xse_resnext50_deeper, act_cls=Mish, sa=True)
        self.model = create_unet_model(
            arch=arch,
            img_size=input_size,
            n_out=forecast_steps,
            pretrained=pretrained,
            n_in=history_steps,
            self_attention=True,
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)
