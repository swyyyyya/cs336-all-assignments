"""Text generation from a trained TransformerLM (CS336 assignment 1, section 6).

Implements autoregressive decoding with optional temperature scaling and
top-p (nucleus) sampling:

    prompt -> tokenize -> model forward -> sample next token from the last
    position's distribution -> append -> repeat until <|endoftext|> or the
    maximum number of generated tokens is reached.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .softmax import softmax


def _apply_temperature_and_top_p(
    logits: Tensor,
    temperature: float,
    top_p: float | None,
) -> Tensor:
    """Turn raw logits into a (renormalized) sampling distribution.

    Args:
        logits (Float[Tensor, " vocab_size"]): Unnormalized logits for the
            next-token distribution at one position.
        temperature (float): Softmax temperature. temperature -> 0 makes the
            distribution concentrate on the argmax; must be > 0.
        top_p (float | None): Nucleus sampling threshold in (0, 1]. Keep the
            smallest set of tokens whose cumulative probability reaches top_p
            and renormalize over that set; None disables top-p filtering.

    Returns:
        Float[Tensor, " vocab_size"]: The normalized sampling distribution
            (filtered-out tokens have probability 0).
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1] or None")

    logits = logits / temperature
    probs = softmax(logits, dim=-1)

    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Nucleus sampling: keep the smallest prefix of the sorted
        # distribution whose cumulative mass reaches top_p. Equivalently,
        # keep token i iff the cumulative mass strictly before it is below
        # the threshold (this automatically keeps the token that crosses it).
        cumsum_before = cumsum - sorted_probs
        keep = (cumsum_before < top_p) & (sorted_probs > 0)

        kept_probs = sorted_probs[keep]
        kept_probs = kept_probs / kept_probs.sum()

        # Scatter the filtered distribution back to the original vocab order.
        filtered = probs.new_zeros(probs.shape)
        filtered.scatter_(-1, sorted_indices[keep], kept_probs)
        return filtered

    return probs


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    tokenizer,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float | None = 0.9,
    eos_token_id: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """Generate a continuation for a prompt, returning the generated token IDs.

    Args:
        model (torch.nn.Module): Trained TransformerLM (in eval mode).
        prompt_ids (list[int]): Token IDs of the prompt (prefix).
        tokenizer: Tokenizer used to check the EOS token, if provided.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Softmax temperature for sampling.
        top_p (float | None): Nucleus sampling threshold (None disables).
        eos_token_id (int | None): Stop when this token is sampled. If None,
            generation only stops at max_new_tokens.
        device (str): Device to run the model on.

    Returns:
        list[int]: The generated token IDs (excluding the prompt).
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")

    model.eval()
    context_length = model.context_length

    generated: list[int] = []
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Only feed the most recent context_length tokens.
        window = input_ids[:, -context_length:]

        logits = model(window)  # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]  # last position only

        if temperature == 0:
            next_id = int(next_logits.argmax().item())
        else:
            dist = _apply_temperature_and_top_p(next_logits, temperature, top_p)
            next_id = int(torch.multinomial(dist, num_samples=1).item())
        generated.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break

        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=-1,
        )

    return generated
