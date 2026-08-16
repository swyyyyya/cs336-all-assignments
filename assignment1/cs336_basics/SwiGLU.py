import torch
from torch import Tensor, nn

from .Linear import Linear
from .silu import silu


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        gate = silu(self.w1(x))
        value = self.w3(x)
        hidden = gate * value
        return self.w2(hidden)