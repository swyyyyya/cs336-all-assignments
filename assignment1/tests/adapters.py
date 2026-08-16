from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import tokenizer
from cs336_basics.Linear import Linear
from cs336_basics.Embedding import Embedding
from cs336_basics.RMSnorm import RMSNorm
from cs336_basics.silu import silu
from cs336_basics.softmax import softmax
from cs336_basics.RoPE import RoPE
from cs336_basics.SwiGLU import SwiGLU
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.multihead_self_attention import multihead_self_attention
from cs336_basics.multihead_self_attention_with_rope import multihead_self_attention_with_rope
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.training import (
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定 Linear 层的权重，计算批量输入经过线性变换后的结果。

    Args:
        in_dim (int): 输入维度大小
        out_dim (int): 输出维度大小
        weights (Float[Tensor, "d_out d_in"]): 使用的线性层权重
        in_features (Float[Tensor, "... d_in"]): 要施加线性变换的输入张量

    Returns:
        Float[Tensor, "... d_out"]: 线性模块变换后的输出。
    """
    linear = Linear(d_in,d_out)
    linear.load_state_dict({"weight": weights})
    return linear(in_features)
    
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 Embedding 层的权重，按一批 token id 取出对应的 embedding。

    Args:
        vocab_size (int): 词表中 embedding 的数量
        d_model (int): embedding 维度大小
        weights (Float[Tensor, "vocab_size d_model"]): 可查询的 embedding 向量
        token_ids (Int[Tensor, "..."]): 要从 Embedding 层取出的 token id

    Returns:
        Float[Tensor, "... d_model"]: Embedding 层返回的批量 embedding。
    """
    embedding = Embedding(vocab_size,d_model)

    with torch.no_grad():
        embedding.weight.copy_(weights)

    return embedding(token_ids)
    
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 SwiGLU 网络的权重，返回你用这些权重实现的输出。

    Args:
        d_model (int): 前馈网络输入与输出的维度。
        d_ff (int): SwiGLU 内部上投影的维度。
        w1_weight (Float[Tensor, "d_ff d_model"]): W1 的权重
        w2_weight (Float[Tensor, "d_model d_ff"]): W2 的权重
        w3_weight (Float[Tensor, "d_ff d_model"]): W3 的权重
        in_features (Float[Tensor, "... d_model"]): 送入前馈层的输入 embedding。

    Returns:
        Float[Tensor, "... d_model"]: 与输入 embedding 同形状的输出 embedding。
    """
    # 1. 初始化SwiGLU模块
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)

    # 2. 手动覆盖权重（Linear.weight 形状正好是 [out_dim, in_dim] 和传入一致）
    swiglu.w1.weight.data = w1_weight
    swiglu.w2.weight.data = w2_weight
    swiglu.w3.weight.data = w3_weight

    # 3. 前向计算并返回结果
    return swiglu(in_features)
    
def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定 key (K)、query (Q)、value (V) 张量，返回你实现的
    scaled dot-product attention 的输出。

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query 张量
        K (Float[Tensor, " ... keys d_k"]): Key 张量
        V (Float[Tensor, " ... keys d_v"]): Value 张量
        mask (Bool[Tensor, " ... queries keys"] | None): 掩码张量
    Returns:
        Float[Tensor, " ... queries d_v"]: SDPA 的输出
    """
    return scaled_dot_product_attention(Q,K,V,mask)

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定朴素、非批量多头注意力的 Q/K/V 投影权重，返回优化后的批量实现输出。
    该实现应在一次矩阵乘法中完成所有头的 Q、K、V 投影。
    此函数不应使用 RoPE。
    参见 Vaswani et al., 2017 第 3.2.2 节。

    Args:
        d_model (int): 前馈输入与输出的维度。
        num_heads (int): 多头注意力的头数。
        max_seq_len (int): 若你的实现会预缓存，则为最大序列长度。
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 要运行实现的输入张量。

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: 使用给定 QKV 投影权重与输入特征，
        运行你的优化批量多头注意力后的输出张量。
    """
    attention = multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
    )
    with torch.no_grad():
        attention.q_proj.weight.copy_(q_proj_weight)
        attention.k_proj.weight.copy_(k_proj_weight)
        attention.v_proj.weight.copy_(v_proj_weight)
        attention.output_proj.weight.copy_(o_proj_weight)
    return attention(in_features)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定朴素、非批量多头注意力的 Q/K/V 投影权重，返回优化后的批量实现输出。
    该实现应在一次矩阵乘法中完成所有头的 Q、K、V 投影。
    本版本的 MHA 应包含 RoPE。
    此时 RoPE 的嵌入维度必须是每个头的嵌入维度（d_model // num_heads）。
    参见 Vaswani et al., 2017 第 3.2.2 节。

    Args:
        d_model (int): 前馈输入与输出的维度。
        num_heads (int): 多头注意力的头数。
        max_seq_len (int): 若你的实现会预缓存，则为最大序列长度。
        theta (float): RoPE 参数。
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出投影权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 要运行实现的输入张量。
        token_positions (Int[Tensor, " ... sequence_length"] | None): 可选，token 位置张量

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: 使用给定 QKV 投影权重与输入特征，
        运行你的优化批量多头注意力后的输出张量。
    """
    attention = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
    )
    with torch.no_grad():
        attention.q_proj.weight.copy_(q_proj_weight)
        attention.k_proj.weight.copy_(k_proj_weight)
        attention.v_proj.weight.copy_(v_proj_weight)
        attention.output_proj.weight.copy_(o_proj_weight)
    return attention(in_features, token_positions)

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对给定输入张量运行 RoPE。

    Args:
        d_k (int): query 或 key 张量的嵌入维度。
        theta (float): RoPE 参数。
        max_seq_len (int): 若你的实现会预缓存，则为最大序列长度。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 要施加 RoPE 的输入张量。
        token_positions (Int[Tensor, "... sequence_length"]): 形状为 (batch_size, sequence_length) 的 token 位置张量
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: 施加 RoPE 后的张量。
    """

    rope = RoPE(theta,d_k,max_seq_len)
    return rope(in_query_or_key,token_positions)

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定 pre-norm Transformer block 的权重与输入特征，
    返回在该输入上运行 Transformer block 的输出。

    此函数应使用 RoPE。
    按你的实现方式，可能只需把相关参数传给 TransformerBlock 构造函数，
    也可能需要自己初始化 RoPE 类再传入。

    Args:
        d_model (int): Transformer block 输入的维度。
        num_heads (int): 多头注意力的头数。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈内层维度。
        max_seq_len (int): 若你的实现会预缓存，则为最大序列长度。
        theta (float): RoPE 参数。
        weights (dict[str, Tensor]):
            参考实现的 state dict。
            该字典的键为：
            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的 query 投影。
                形状为 (d_model, d_model)。
                按形状 (num_heads, d_k) 的矩阵沿行拼接，
                因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 个注意力头的 key 投影。
                形状为 (d_model, d_model)。
                按形状 (num_heads, d_k) 的矩阵沿行拼接，
                因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有 `num_heads` 个注意力头的 value 投影。
                形状为 (d_model, d_model)。
                按形状 (num_heads, d_v) 的矩阵沿行拼接，
                因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影的权重
                形状为 (d_model, d_model)。
            - `ln1.weight`
                Transformer block 中第一个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `ffn.w1.weight`
                FFN 中第一次线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ffn.w2.weight`
                FFN 中第二次线性变换的权重。
                形状为 (d_model, d_ff)。
            - `ffn.w3.weight`
                FFN 中第三次线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ln2.weight`
                Transformer block 中第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            要运行实现的输入张量。

    Returns:
        Float[Tensor, "batch sequence_length d_model"]：在输入特征上运行
        带 RoPE 的 Transformer block 后的输出。
    """
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
    )
    block.load_state_dict(weights)
    return block(in_features)

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定 Transformer 语言模型的权重与输入索引，
    返回在输入索引上前向传播的输出。

    此函数应使用 RoPE。

    Args:
        vocab_size (int): 要预测的输出词表中不重复项的数量。
        context_length (int): 一次最多处理的 token 数。
        d_model (int): 模型 embedding 与子层输出的维度。
        num_layers (int): 使用的 Transformer 层数。
        num_heads (int): 多头注意力的头数。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈内层维度（第 3.3 节）。
        rope_theta (float): RoPE 的 $\\Theta$ 参数。
        weights (dict[str, Tensor]):
            参考实现的 state dict。{num_layers} 表示
            `0` 到 `num_layers - 1` 之间的整数（层索引）。
            该字典的键为：
            - `token_embeddings.weight`
                Token embedding 矩阵。形状为 (vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`
                所有 `num_heads` 个注意力头的 query 投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                按形状 (num_heads, d_k) 的矩阵沿行拼接，
                因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.k_proj.weight`
                所有 `num_heads` 个注意力头的 key 投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                按形状 (num_heads, d_k) 的矩阵沿行拼接，
                因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.v_proj.weight`
                所有 `num_heads` 个注意力头的 value 投影。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                按形状 (num_heads, d_v) 的矩阵沿行拼接，
                因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.output_proj.weight`
                多头自注意力输出投影的权重
                形状为 ((d_model / num_heads) * num_heads, d_model)。
            - `layers.{num_layers}.ln1.weight`
                Transformer block 中第一个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `layers.{num_layers}.ffn.w1.weight`
                FFN 中第一次线性变换的权重。
                形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ffn.w2.weight`
                FFN 中第二次线性变换的权重。
                形状为 (d_model, d_ff)。
            - `layers.{num_layers}.ffn.w3.weight`
                FFN 中第三次线性变换的权重。
                形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ln2.weight`
                Transformer block 中第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `ln_final.weight`
                作用于最后一个 transformer block 输出的 RMSNorm 仿射变换权重。
                形状为 (d_model, )。
            - `lm_head.weight`
                语言模型输出 embedding 的权重。
                形状为 (vocab_size, d_model)。
        in_indices (Int[Tensor, "batch_size sequence_length"])：要送入语言模型的输入索引。
            形状为 (batch_size, sequence_length)，其中 `sequence_length` 至多为 `context_length`。

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: 每个 token 的
        未归一化下一词分布预测。
    """
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    model.load_state_dict(weights)
    return model(in_indices)

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 RMSNorm 仿射变换的权重，
    返回在输入特征上运行 RMSNorm 的输出。

    Args:
        d_model (int): RMSNorm 输入的维度。
        eps: (float): 为数值稳定性加到分母上的值。
        weights (Float[Tensor, "d_model"]): RMSNorm 权重。
        in_features (Float[Tensor, "... d_model"]): 要运行 RMSNorm 的输入特征。
            可以有任意前置维度。

    Returns:
        Float[Tensor,"... d_model"]: 与 `in_features` 同形状的张量，
        为对 `in_features` 运行 RMSNorm 的结果。
    """
    rmsnorm = RMSNorm(d_model)
    with torch.no_grad():
        rmsnorm.weight.copy_(weights)
    return rmsnorm(in_features)

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """给定输入张量，返回对每个元素施加 SiLU 后的输出。

    Args:
        in_features(Float[Tensor, "..."]): 要运行 SiLU 的输入特征。形状任意。

    Returns:
        Float[Tensor,"..."]: 与 `in_features` 同形状，为对每个元素施加 SiLU 的结果。
    """
    return silu(in_features)

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定数据集（一维整数 numpy 数组）以及期望的 batch size 与 context length，
    从数据集中采样语言模型的输入序列及其对应标签。

    Args:
        dataset (np.array): 数据集中的一维整数 token ID 数组。
        batch_size (int): 期望采样的 batch 大小。
        context_length (int): 每个采样样本的期望上下文长度。
        device (str): PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），
            表示将采样得到的输入序列与标签放到哪个设备上。

    Returns:
        形状均为 (batch_size, context_length) 的 torch.LongTensor 二元组。
        第一项是采样的输入序列，第二项是对应的语言建模标签。
    """
    return get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定输入张量，返回在指定 `dim` 上做 softmax 的结果。

    Args:
        in_features (Float[Tensor, "..."]): 要做 softmax 的输入特征。形状任意。
        dim (int): 对 `in_features` 施加 softmax 的维度。

    Returns:
        Float[Tensor, "..."]: 与 `in_features` 同形状，为在指定 `dim` 上
        softmax 归一化后的结果。
    """
    return softmax(in_features,dim)

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """给定输入与目标张量，计算各样本的平均交叉熵损失。

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] 是
            第 i 个样本第 j 类的未归一化 logit。
        targets (Int[Tensor, "batch_size"]): 形状为 (batch_size,) 的张量，为正确类别的索引。
            每个值必须在 0 到 `num_classes - 1` 之间。

    Returns:
        Float[Tensor, ""]: 各样本的平均交叉熵损失。
    """
    return cross_entropy(inputs,targets)

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """给定一组参数，将其梯度整体裁剪，使 L2 范数至多为 max_l2_norm。

    Args:
        parameters (Iterable[torch.nn.Parameter]): 可训练参数集合。
        max_l2_norm (float): 正数，表示允许的最大 L2 范数。

    应原地修改参数的梯度（parameter.grad）。
    """
    gradient_clipping(parameters, max_l2_norm)

def get_adamw_cls() -> Any:
    """
    返回一个实现了 AdamW 的 torch.optim.Optimizer 类。
    """
    return AdamW

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定带线性 warmup 的余弦学习率衰减日程参数以及迭代次数，
    返回该日程下指定迭代的学习率。

    Args:
        it (int): 要查询学习率的迭代编号。
        max_learning_rate (float): alpha_max，余弦学习率日程（含 warmup）的最大学习率。
        min_learning_rate (float): alpha_min，余弦学习率日程（含 warmup）的最小/最终学习率。
        warmup_iters (int): T_w，线性 warmup 学习率的迭代次数。
        cosine_cycle_iters (int): T_c，余弦退火的迭代次数。

    Returns:
        该日程下指定迭代的学习率。
    """
    return get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器与迭代次数，将它们序列化到磁盘。

    Args:
        model (torch.nn.Module): 序列化该模型的状态。
        optimizer (torch.optim.Optimizer): 序列化该优化器的状态。
        iteration (int): 序列化该值，表示已完成的训练迭代次数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 用于写入模型、优化器与迭代次数的路径或类文件对象。
    """
    save_checkpoint(model, optimizer, iteration, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    给定序列化的 checkpoint（路径或类文件对象），将状态恢复到给定模型与优化器。
    返回此前在 checkpoint 中序列化的迭代次数。

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化 checkpoint 的路径或类文件对象。
        model (torch.nn.Module): 恢复该模型的状态。
        optimizer (torch.optim.Optimizer): 恢复该优化器的状态。
    Returns:
        int: 此前序列化的迭代次数。
    """
    return load_checkpoint(src, model, optimizer)

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """给定词表、merge 列表与特殊 token 列表，
    返回使用这些 vocab、merges 与 special tokens 的 BPE tokenizer。

    Args:
        vocab (dict[int, bytes]): tokenizer 词表，从 int（词表中的 token ID）
            映射到 bytes（token 字节）
        merges (list[tuple[bytes, bytes]]): BPE merges。每一项是 bytes 二元组 (<token1>, <token2>)，
            表示 <token1> 与 <token2> 发生了合并。
            merges 按创建顺序排列。
        special_tokens (list[str] | None): tokenizer 的字符串特殊 token 列表。这些字符串永远不会
            被拆成多个 token，始终保持为单个 token。

    Returns:
        使用所提供 vocab、merges 与 special tokens 的 BPE tokenizer。
    """
    return tokenizer(vocab, merges, special_tokens)
    
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """给定输入语料路径，训练 BPE tokenizer，
    并输出其词表与 merges。

    Args:
        input_path (str | os.PathLike): BPE tokenizer 训练数据的路径。
        vocab_size (int): tokenizer 词表总大小（包含 special tokens）。
        special_tokens (list[str]): 要加入 tokenizer 词表的字符串特殊 token 列表。
            这些字符串永远不会被拆成多个 token，始终保持为单个 token。
            若它们出现在 `input_path` 中，按普通字符串处理。

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练得到的 tokenizer 词表，从 int（词表中的 token ID）
                映射到 bytes（token 字节）
            merges:
                BPE merges。每一项是 bytes 二元组 (<token1>, <token2>)，
                表示 <token1> 与 <token2> 发生了合并。
                merges 按创建顺序排列。
    """
    return train_bpe(input_path,vocab_size,special_tokens)
    
