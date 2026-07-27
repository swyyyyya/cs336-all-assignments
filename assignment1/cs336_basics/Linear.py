import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        # 权重W：【out_features, in_features】
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        # 讲义指定初始化：trunc_normal_
        torch.nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [..., in_features]
        return shape: [..., out_features]
        等价 x @ W.T
        """
        return torch.einsum("...i,oi->...o", x, self.weight)