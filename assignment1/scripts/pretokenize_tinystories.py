"""Tokenize a raw text corpus into a .npy file of token IDs.

The produced .npy file can be loaded by the training script as a
memory-mapped array (``np.load(..., mmap_mode='r')``), so corpora much
larger than RAM can be handled without loading them fully into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Raw text corpus to tokenize",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to write the .npy file of token IDs",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        required=True,
        help="BPE vocab.json (GPT-2 style serialization)",
    )
    parser.add_argument(
        "--merges-path",
        type=Path,
        required=True,
        help="BPE merges.txt (GPT-2 style serialization)",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="Special token string (repeatable). Defaults to ['<|endoftext|>'].",
    )
    parser.add_argument(
        "--dtype",
        default="uint16",
        choices=["uint16", "int32"],
        help="Numpy dtype for the token IDs (uint16 fits vocabs up to 65535).",
    )
    args = parser.parse_args()

    special_tokens = args.special_token or ["<|endoftext|>"]
    tok = tokenizer.from_files(
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_tokens=special_tokens,
    )

    if args.dtype == "uint16" and len(tok.vocab) > 2**16:
        raise ValueError(
            f"vocab size {len(tok.vocab)} does not fit in uint16; use --dtype int32"
        )

    token_ids: list[int] = []
    with args.input_path.open(encoding="utf-8") as f:
        for line in f:
            token_ids.extend(tok.encode(line))

    arr = np.array(token_ids, dtype=np.dtype(args.dtype))
    np.save(args.output_path, arr)

    print(f"tokenized {args.input_path} -> {args.output_path}")
    print(f"  tokens: {len(arr):,}")
    print(f"  size on disk: {arr.nbytes / 1024 ** 2:.1f} MiB")


if __name__ == "__main__":
    main()
