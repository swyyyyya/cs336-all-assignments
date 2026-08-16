import torch
from torch import Tensor, nn

from .Embedding import Embedding
from .Linear import Linear
from .RMSnorm import RMSNorm
from .TransformerBlock import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        sequence_length = token_ids.shape[-1]

        if sequence_length > self.context_length:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
            )

        token_positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(
                x,
                token_positions=token_positions,
            )

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits