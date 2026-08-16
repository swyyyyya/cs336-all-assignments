from torch import Tensor, nn
from .RMSnorm import RMSNorm
from .SwiGLU import SwiGLU
from .multihead_self_attention_with_rope import multihead_self_attention_with_rope


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = multihead_self_attention_with_rope(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(
            self.ln1(x),
            token_positions=token_positions,
        )
        x = x + self.ffn(self.ln2(x))
        return x