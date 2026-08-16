import math
import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy as np
import torch
from torch import Tensor
from torch import nn


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """Sample a batch of (input, target) token sequences from a token dataset.

    Args:
        dataset (np.ndarray): 1-D integer array of token IDs.
        batch_size (int): Number of sequences to sample.
        context_length (int): Length of each sampled sequence.
        device (str): Device string (e.g. 'cpu', 'cuda:0') for the tensors.

    Returns:
        A pair of (batch_size, context_length) LongTensors on `device`:
        the input sequences and the corresponding next-token targets
        (each input shifted by one position).
    """
    starting_indices = torch.randint(
        len(dataset) - context_length,
        (batch_size,),
    )
    x = torch.stack(
        [
            torch.from_numpy(dataset[i : i + context_length].astype(np.int64))
            for i in starting_indices
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(dataset[i + 1 : i + 1 + context_length].astype(np.int64))
            for i in starting_indices
        ]
    )
    return x.to(device), y.to(device)


def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip the global L2 norm of the parameter gradients in place.

    If the total L2 norm of all gradients exceeds `max_l2_norm`, every
    gradient is scaled down by max_l2_norm / (total_norm + 1e-6); otherwise
    the gradients are left untouched. A small epsilon (1e-6, the PyTorch
    default) is added to the denominator for numerical stability.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach(), 2) for g in grads]),
        2,
    )
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.mul_(clip_coef)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the learning rate at iteration `it` under a linear-warmup-then-
    cosine-annealing schedule (LLaMA-style).

    Args:
        it (int): Current iteration.
        max_learning_rate (float): Peak learning rate reached at the end of warmup.
        min_learning_rate (float): Final learning rate after annealing.
        warmup_iters (int): T_w, number of linear-warmup iterations.
        cosine_cycle_iters (int): T_c, the final iteration of cosine annealing.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_factor * (
            max_learning_rate - min_learning_rate
        )
    return min_learning_rate


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    config: dict | None = None,
) -> None:
    """Serialize the model, optimizer state, and iteration count to `out`.

    Args:
        config (dict | None): Optional extra metadata (e.g. model
            hyperparameters) stored under the ``"config"`` key.
    """
    checkpoint: dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if config is not None:
        checkpoint["config"] = config
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore `model` and `optimizer` from a checkpoint at `src`.

    Returns the iteration number that was saved in the checkpoint.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
