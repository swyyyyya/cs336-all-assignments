import torch
from einops import rearrange
from torch import Tensor, nn

from .Linear import Linear
from .scaled_dot_product_attention import scaled_dot_product_attention
from .RoPE import RoPE

class multihead_self_attention_with_rope(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.rope = RoPE(
            theta=theta,
            d_k=self.head_dim,
            max_seq_len=max_seq_len,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        #获取序列长度
        sequence_length = x.shape[-2]

        #默认位置序列是 0 ~ L-1
        if token_positions is None:
            token_positions = torch.arange(
                sequence_length,
                device=x.device,
            )

        #线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        #拆分多头
        q = rearrange(
            q,
            "... seq (head d) -> ... head seq d",
            head=self.num_heads,
        )
        k = rearrange(
            k,
            "... seq (head d) -> ... head seq d",
            head=self.num_heads,
        )
        v = rearrange(
            v,
            "... seq (head d) -> ... head seq d",
            head=self.num_heads,
        )

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            )
        )

        attention_output = scaled_dot_product_attention(
            q,
            k,
            v,
            causal_mask
        )

        attention_output = rearrange(
            attention_output,
            "... head seq d -> ... seq (head d)",
        )

        return self.output_proj(attention_output)