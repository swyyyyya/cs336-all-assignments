import math
import torch
from torch import Tensor
from .softmax import softmax

def scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor| None=None
)->Tensor:
    #取最后一维
    d_k = q.shape[-1]

    scores=q@ k.transpose(-2,-1)
    scores=scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask,float("-inf"))

    attention_weights=softmax(scores,dim=-1)

    return attention_weights@v