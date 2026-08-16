"""Generate text continuations from a trained checkpoint.

Example:

    uv run python scripts/generate.py \
        --checkpoint outputs/checkpoints/checkpoint_005000.pt \
        --vocab-path outputs/tinystories_train_10k/vocab.json \
        --merges-path outputs/tinystories_train_10k/merges.txt \
        --prompt "Once upon a time" --max-tokens 200 --temperature 0.8 --top-p 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.decoding import generate
from cs336_basics.tokenizer import tokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model checkpoint (.pt)")
    parser.add_argument("--vocab-path", type=Path, required=True, help="BPE vocab.json")
    parser.add_argument("--merges-path", type=Path, required=True, help="BPE merges.txt")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to continue from")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy argmax")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus threshold; 0 disables it")
    parser.add_argument("--device", type=str, default=None, help="cpu / mps / cuda (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--special-token", action="append", default=[], help="Special tokens (default: <|endoftext|>)")
    return parser.parse_args()


def _pick_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = _pick_device(args.device)

    special_tokens = args.special_token or ["<|endoftext|>"]
    tok = tokenizer.from_files(
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_tokens=special_tokens,
    )
    eos_token_id = tok.str_to_id.get("<|endoftext|>")

    # Load only the model state from the checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    # Reconstruct the model with the hyperparameters it was trained with
    # (stored in the checkpoint by the training script).
    config = (
        checkpoint.get("config", {})
        if isinstance(checkpoint, dict)
        else {}
    )
    model = TransformerLM(
        vocab_size=config.get("vocab_size", len(tok.vocab)),
        context_length=config.get("context_length", 256),
        d_model=config.get("d_model", 512),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 16),
        d_ff=config.get("d_ff", 1344),
        rope_theta=config.get("rope_theta", 10_000.0),
    )
    model.load_state_dict(state_dict)
    model.to(device)

    prompt_ids = tok.encode(args.prompt)
    print(f"prompt ids ({len(prompt_ids)} tokens): {prompt_ids[:20]}{'...' if len(prompt_ids) > 20 else ''}")

    generated_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        tokenizer=tok,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p if args.top_p > 0 else None,
        eos_token_id=eos_token_id,
        device=device,
    )

    print("\n" + "=" * 60)
    print(f"PROMPT: {args.prompt}")
    print("-" * 60)
    print(args.prompt + tok.decode(generated_ids))
    print("=" * 60)


if __name__ == "__main__":
    main()
