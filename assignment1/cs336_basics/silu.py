import torch
from jaxtyping import Float

def silu(x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
        out = x / (1.0 + torch.exp(-x))
        return out