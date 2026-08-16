import torch
from torch import Tensor, nn


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        # RoPE要求特征维度必须为偶数，两两分组做二维旋转变换
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # RoPE 基础频率公式：f = θ^(-2i / d_k)
        pair_indices = torch.arange(0, d_k, 2, device=device)
        inv_freq = theta ** (-pair_indices / d_k)

        positions = torch.arange(max_seq_len, device=device)
        # 广播：[seq_len, 1] * [1, d_k//2] → [max_seq_len, d_k//2]
        # angles[pos, i] = pos * f_i 每个位置、每一组的旋转弧度
        angles = positions[:, None] * inv_freq[None, :]

         # 预计算cos、sin值，注册为buffer：不参与梯度更新、不存入模型权重文件
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor,
    ) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"Expected final dimension {self.d_k}, got {x.shape[-1]}"
            )

        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                "token_positions and x must have the same sequence length"
            )

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        #取词向量x中的偶数位置元素
        x_even = x[..., 0::2]
        #取词向量x中的奇数位置元素
        x_odd = x[..., 1::2]

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        # 二维旋转矩阵作用：
        # [cos  -sin] [x_even]
        # [sin   cos] [x_odd]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        #拼接
        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(-2)

if __name__ == "__main__":
    rope = RoPE(theta=10000.0, d_k=4, max_seq_len=10)
    # batch=1, head=1, seq_len=3, d_k=4
    x = torch.randn(1, 1, 3, 4)
    pos = torch.tensor([0, 1, 2])
    res = rope(x, pos)
    print(res.shape)  # torch.Size([1, 1, 3, 4])