import torch
from torch import Tensor
from jaxtyping import Float
from jaxtyping import Int
from .softmax import softmax

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:

    max_vals = inputs.max(dim=-1, keepdim=True).values   # (batch_size, 1)
    shifted = inputs - max_vals                           # 最大值变为 0，exp 不溢出
    log_sum_exp = max_vals.squeeze(-1) + shifted.exp().sum(dim=-1).log()  # (batch_size,)

    # 从 inputs 中取出每个样本 target 对应的 logit
    bsz = inputs.shape[0]
    idx = torch.arange(bsz, device=inputs.device)
    target_logits = inputs[idx, targets]                  # (batch_size,)

    #负均值得到损失
    return (log_sum_exp - target_logits).mean()