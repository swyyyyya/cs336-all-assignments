import torch
from einops import rearrange
from torch import Tensor, nn

from .Linear import Linear
from .scaled_dot_product_attention import scaled_dot_product_attention


class multihead_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

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

        sequence_length = x.shape[-2]

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            )
        )

        attention_output = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=causal_mask,
        )

        attention_output = rearrange(
            attention_output,
            "... head seq d -> ... seq (head d)",
        )

        return self.output_proj(attention_output)