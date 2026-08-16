import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        # 权重W：【out_features, in_features】
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        # 讲义指定初始化：N(0, 2/(d_in+d_out))，截断于 [-3σ, 3σ]
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [..., in_features]
        return shape: [..., out_features]
        等价 x @ W.T
        """
        return torch.einsum("...i,oi->...o", x, self.weight)