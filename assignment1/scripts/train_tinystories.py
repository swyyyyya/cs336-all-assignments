"""Train a TransformerLM with AdamW on a pretokenized (memmap) corpus.

This is the main training loop for the CS336 assignment 1 ``training_together``
problem. It:

* loads token-ID corpora as numpy memory-mapped arrays (``mmap_mode='r'``),
* samples batches with ``get_batch``,
* trains with AdamW under a linear-warmup + cosine-annealing LR schedule,
* clips gradients, and periodically
* evaluates validation loss/perplexity,
* logs progress (console and optionally Weights & Biases),
* serializes resumable checkpoints.

Example (low-resource TinyStories run, ~40M tokens):

    uv run python scripts/train_tinystories.py \
        --train-data outputs/tinystories_train_tokens.npy \
        --valid-data outputs/tinystories_valid_tokens.npy \
        --total-steps 5000 --device mps
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.training import (
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data
    parser.add_argument("--train-data", type=Path, required=True, help=".npy token-ID corpus (memmap)")
    parser.add_argument("--valid-data", type=Path, required=True, help=".npy token-ID corpus (memmap)")

    # Model (recommended TinyStories settings from the handout)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    # Optimizer / LR schedule
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (alpha_max)")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Final learning rate (alpha_min)")
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument(
        "--cosine-cycle-iters",
        type=int,
        default=None,
        help="Final cosine-annealing iteration T_c (defaults to --total-steps)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    # Training loop
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=100,
        help="Number of sampled validation batches per eval; 0 = full pass over the validation set",
    )
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="JSONL file to append training metrics to (default: <save-dir>/metrics.jsonl)",
    )

    # Runtime
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu / mps / cuda (default: auto-detect)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="JIT-compile the model with torch.compile")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="cs336-assignment1")

    return parser.parse_args()


def _pick_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_dataset(path: Path, vocab_size: int) -> np.ndarray:
    # mmap_mode='r' keeps the corpus on disk and lazily pages it into memory.
    data = np.load(path, mmap_mode="r")
    if data.max() >= vocab_size:
        raise ValueError(
            f"token id {data.max()} in {path} exceeds vocab_size {vocab_size}; "
            "did you pretokenize with a matching tokenizer?"
        )
    return data


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    valid_data: np.ndarray,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str,
    num_batches: int,
) -> float:
    """Average per-token cross-entropy over the validation set.

    With num_batches > 0, that many batches are sampled at random; with
    num_batches == 0, a full deterministic pass over the data is made.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    if num_batches == 0:
        # Full pass: only full batches, no reshaping surprises at the tail.
        num_start_positions = len(valid_data) - context_length
        step = batch_size * context_length
        for start in range(0, num_start_positions - step + 1, step):
            end = start + step
            x = torch.from_numpy(valid_data[start:end].astype(np.int64)).reshape(
                batch_size, context_length
            ).to(device)
            y = torch.from_numpy(valid_data[start + 1 : end + 1].astype(np.int64)).reshape(
                batch_size, context_length
            ).to(device)
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item() * step
            total_tokens += step
    else:
        for _ in range(num_batches):
            x, y = get_batch(valid_data, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item() * (batch_size * context_length)
            total_tokens += batch_size * context_length

    model.train()
    if total_tokens == 0:
        # Validation set too small for even one batch (e.g. full-pass mode).
        return float("nan")
    return total_loss / total_tokens


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _pick_device(args.device)
    print(f"device: {device}", flush=True)

    train_data = _load_dataset(args.train_data, args.vocab_size)
    valid_data = _load_dataset(args.valid_data, args.vocab_size)
    print(
        f"train tokens: {len(train_data):,} | valid tokens: {len(valid_data):,}",
        flush=True,
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    if args.compile:
        # On MPS the aot_eager backend is the recommended torch.compile path.
        backend = "aot_eager" if device == "mps" else None
        model = torch.compile(model, backend=backend)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    cosine_cycle_iters = args.cosine_cycle_iters or args.total_steps
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    if args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))

    args.save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_path or (args.save_dir / "metrics.jsonl")
    metrics_file = open(metrics_path, "a", encoding="utf-8")

    # Model hyperparameters are stored in checkpoints so that e.g. the
    # generation script can reconstruct the model without re-passing them.
    checkpoint_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }

    model.train()
    t_start = time.perf_counter()
    tokens_seen = 0

    for step in range(start_step, args.total_steps):
        # Set the learning rate for this step according to the schedule.
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        tokens_seen += args.batch_size * args.context_length
        step_num = step + 1

        if step_num % args.log_interval == 0 or step_num == args.total_steps:
            elapsed = time.perf_counter() - t_start
            tokens_per_sec = tokens_seen / elapsed
            print(
                f"step {step_num}/{args.total_steps} | lr {lr:.3e} | "
                f"train loss {loss.item():.4f} | {tokens_per_sec:,.0f} tok/s | "
                f"{elapsed / 60:.1f} min",
                flush=True,
            )
            metrics_file.write(
                json.dumps(
                    {
                        "step": step_num,
                        "lr": lr,
                        "train_loss": loss.item(),
                        "tokens_per_sec": tokens_per_sec,
                        "elapsed_sec": elapsed,
                    }
                )
                + "\n"
            )
            metrics_file.flush()
            if args.wandb:
                wandb.log(
                    {
                        "step": step_num,
                        "lr": lr,
                        "train_loss": loss.item(),
                        "tokens_per_sec": tokens_per_sec,
                    }
                )

        if step_num % args.eval_interval == 0 or step_num == args.total_steps:
            val_loss = evaluate(
                model=model,
                valid_data=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                vocab_size=args.vocab_size,
                device=device,
                num_batches=args.eval_batches,
            )
            val_ppl = math.exp(val_loss)
            print(f"  eval: val loss {val_loss:.4f} | val ppl {val_ppl:.2f}", flush=True)
            metrics_file.write(
                json.dumps(
                    {
                        "step": step_num,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                    }
                )
                + "\n"
            )
            metrics_file.flush()
            if args.wandb:
                wandb.log({"step": step_num, "val_loss": val_loss, "val_ppl": val_ppl})

        if step_num % args.save_interval == 0:
            ckpt_path = args.save_dir / f"checkpoint_{step_num:06d}.pt"
            save_checkpoint(model, optimizer, step_num, ckpt_path, config=checkpoint_config)
            print(f"  saved {ckpt_path}", flush=True)

    final_path = args.save_dir / f"checkpoint_{args.total_steps:06d}.pt"
    save_checkpoint(model, optimizer, args.total_steps, final_path, config=checkpoint_config)
    print(f"saved final checkpoint to {final_path}", flush=True)
    print(f"metrics logged to {metrics_path}", flush=True)
    metrics_file.close()


if __name__ == "__main__":
    main()
