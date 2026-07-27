import torch
from jaxtyping import Float
from torch import Tensor

def softmax(x:Float[Tensor,"..."],dim: int)->Float[Tensor,"..."]:
    #指定维度求最大值，保持数值稳定性
    max_vals = torch.max(x,dim=dim,keepdim=True).values
    #所有值减去最大值
    x_shifted = x - max_vals
    #分子
    exp_x = torch.exp(x_shifted)
    #分母
    sum_exp = torch.sum(exp_x,dim=dim,keepdim=True)
    #返回
    softmax_out = exp_x/sum_exp
    return softmax_out

