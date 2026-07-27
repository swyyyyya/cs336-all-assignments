import torch 
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)

        rms = torch.sqrt(
            torch.mean(x_float.square(), dim=-1, keepdim=True) + self.eps
        )
        normalized = x_float / rms

        return (normalized * self.weight).to(input_dtype)