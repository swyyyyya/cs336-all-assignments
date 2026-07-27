import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(
        self,
        token_ids: Int[Tensor, " ..."],
    ) -> Float[Tensor, " ... embedding_dim"]:
        return self.weight[token_ids]