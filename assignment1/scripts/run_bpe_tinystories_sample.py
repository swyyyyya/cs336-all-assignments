"""Smoke-run BPE training on tinystories_sample_5M.txt (not the full dataset)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import psutil

from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import bytes_to_unicode

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "tests" / "fixtures" / "tinystories_sample_5M.txt"
OUT_DIR = ROOT / "outputs" / "tinystories_sample_5M"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]


def main() -> None:
    assert INPUT_PATH.is_file(), f"missing input: {INPUT_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    rss_before = process.memory_info().rss
    peak_rss = rss_before

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )
    elapsed = time.perf_counter() - t0
    peak_rss = max(peak_rss, process.memory_info().rss)

    # Serialize in GPT-2 style: bytes are remapped to printable unicode
    # strings (via bytes_to_unicode), so that space-separated merges.txt lines
    # can be parsed back unambiguously. See tokenizer.from_files.
    byte_encoder = bytes_to_unicode()
    vocab_path = OUT_DIR / "vocab.json"
    merges_path = OUT_DIR / "merges.txt"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                str(i): "".join(byte_encoder[b] for b in token)
                for i, token in vocab.items()
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with merges_path.open("w", encoding="utf-8") as f:
        for a, b in merges:
            a_str = "".join(byte_encoder[x] for x in a)
            b_str = "".join(byte_encoder[x] for x in b)
            f.write(f"{a_str} {b_str}\n")

    longest = max(vocab.values(), key=len)
    print("==== summary ====")
    print(f"input:          {INPUT_PATH}")
    print(f"vocab_size:     {len(vocab)} (target {VOCAB_SIZE})")
    print(f"num_merges:     {len(merges)}")
    print(f"time_sec:       {elapsed:.2f}")
    print(f"rss_before_mb:  {rss_before / (1024 ** 2):.1f}")
    print(f"rss_after_mb:   {peak_rss / (1024 ** 2):.1f}")
    print(f"longest_token:  {longest!r} (len={len(longest)} bytes)")
    print(f"wrote:          {vocab_path}")
    print(f"wrote:          {merges_path}")


if __name__ == "__main__":
    main()
