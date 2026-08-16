"""Full TinyStories BPE training for problem train_bpe_tinystories (a)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import psutil

from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import bytes_to_unicode

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
OUT_DIR = ROOT / "outputs" / "tinystories_train_10k"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]


def main() -> None:
    assert INPUT_PATH.is_file(), f"missing input: {INPUT_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    rss_before = process.memory_info().rss
    peak_rss = rss_before

    print(f"starting train on {INPUT_PATH} ...", flush=True)
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
    print("==== summary ====", flush=True)
    print(f"input:          {INPUT_PATH}", flush=True)
    print(f"vocab_size:     {len(vocab)} (target {VOCAB_SIZE})", flush=True)
    print(f"num_merges:     {len(merges)}", flush=True)
    print(f"time_sec:       {elapsed:.2f}", flush=True)
    print(f"time_min:       {elapsed / 60:.2f}", flush=True)
    print(f"rss_before_mb:  {rss_before / (1024 ** 2):.1f}", flush=True)
    print(f"rss_after_mb:   {peak_rss / (1024 ** 2):.1f}", flush=True)
    print(f"longest_token:  {longest!r} (len={len(longest)} bytes)", flush=True)
    print(f"wrote:          {vocab_path}", flush=True)
    print(f"wrote:          {merges_path}", flush=True)


if __name__ == "__main__":
    main()
