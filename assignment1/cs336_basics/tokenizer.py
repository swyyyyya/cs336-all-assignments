from typing import Iterator, Optional
import regex

class tokenizer:
    # 预分词正则
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        for s in self.special_tokens:
            b=s.encode("utf-8")
            if b not in self.vocab.values():
                self.vocab[len(self.vocab)] = b
      
        # 构建反向映射 bytes -> token id
        self.bytes_to_id: dict[bytes, int] = {b: idx for idx, b in vocab.items()}
        # 构建merge优先级映射
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair:i for i,pair in enumerate(merges)}
        self.str_to_id: dict[str, int] = {s: self.bytes_to_id[s.encode("utf-8")] for s in self.special_tokens}

    def _bpe_merge(self, token_bytes: list[bytes]) -> list[bytes]:
        while len(token_bytes) >= 2:
            # 寻找可合并、rank最小的pair
            min_rank = None
            target_pair = None
            for i in range(len(token_bytes)-1):
                pair = (token_bytes[i], token_bytes[i+1])
                if pair in self.merge_ranks:
                    rk = self.merge_ranks[pair]
                    if (min_rank is None) or rk < min_rank:
                        min_rank = rk
                        target_pair = (i, pair)
            if target_pair is None:
                break
            idx, (a,b) = target_pair
            new_seq = []
            i = 0
            while i < len(token_bytes):
                if i == idx and token_bytes[i]==a and token_bytes[i+1]==b:
                    new_seq.append(a + b)
                    i += 2
                else:
                    new_seq.append(token_bytes[i])
                    i += 1
            token_bytes = new_seq
        return token_bytes

    def encode(self, text: str) -> list[int]:
        token_ids = []
        # 长token优先匹配，避免短special抢占长special前缀
        sorted_specials = sorted(self.special_tokens, key=lambda s: -len(s))

        ptr = 0
        text_len = len(text)
        while ptr < text_len:
            matched_sp = None
            # 尝试匹配任意完整特殊token
            for sp in sorted_specials:
                sp_len = len(sp)
                if ptr + sp_len <= text_len and text[ptr:ptr+sp_len] == sp:
                    matched_sp = sp
                    break
            if matched_sp is not None:
                # 命中special：直接加入id，跳过这段字符
                token_ids.append(self.str_to_id[matched_sp])
                ptr += len(matched_sp)
            else:
                # 找到下一个special出现的位置，截取【普通文本块】
                next_pos = text_len
                for sp in sorted_specials:
                    pos = text.find(sp, ptr)
                    if pos != -1 and pos < next_pos:
                        next_pos = pos
                plain_chunk = text[ptr:next_pos]
                ptr = next_pos

                # ===== 普通文本：执行原有预分词 + BPE =====
                for match in regex.finditer(self.PAT, plain_chunk):
                    chunk_str = match.group(0)
                    raw_bytes = list(chunk_str.encode("utf-8"))
                    seg = [bytes([b]) for b in raw_bytes]
                    merged_seg = self._bpe_merge(seg)
                    for sub_bytes in merged_seg:
                        token_ids.append(self.bytes_to_id[sub_bytes])
        return token_ids

    def encode_iterable(self, iterable) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_segments = []
        for token_id in ids:
            byte_segments.append(self.vocab[token_id])
        full_bytes = b"".join(byte_segments)
        return full_bytes.decode("utf-8", errors="replace")
    

if __name__ == "__main__":
    vocab = {0: b"h", 1: b"i", 2: b"!"}
    tok = tokenizer(vocab=vocab, merges=[], special_tokens=None)
    print(tok.decode([0, 1, 2]))   
    print(tok.decode([0,266]))    
           