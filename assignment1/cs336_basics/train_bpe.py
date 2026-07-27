import os
import regex

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 定义词汇表
    vocab: dict[int, bytes] = {}

    # 词汇表初始化，256个基础单字节
    for i in range(256):
        vocab[i] = bytes([i])

    # 加入完整特殊token到词表
    for token in special_tokens:
        if len(vocab) >= vocab_size:
            raise ValueError(
                f"目标词表大小 {vocab_size} 过小，无法容纳256基础字节与{len(special_tokens)}个特殊token"
            )
        vocab[len(vocab)] = token.encode('utf-8')
    print(f"初始词表构建完成。基础字节: 256, 特殊Token: {len(special_tokens)}, 当前词表大小: {len(vocab)}")

    # 读取文件内容
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ============【重点修改区域开始】============
    # 使用 special token 作为分隔符切分文本，防止跨special产生pair
    pre_token_counts: dict[str, int] = {}
    if special_tokens:
        # 构造正则：多个special任选其一，自动转义特殊字符如 < |
        escaped = [regex.escape(sp) for sp in special_tokens]
        split_pat = "|".join(escaped)
        chunks = regex.split(split_pat, text)
    else:
        chunks = [text]

    # 每个chunk单独预分词，结果汇总到同一个计数字典
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        if not chunk:
            continue
        for match in regex.finditer(PAT, chunk):
            token_str = match.group(0)
            pre_token_counts[token_str] = pre_token_counts.get(token_str, 0) + 1
    # ============【重点修改区域结束】============

    # 预token转为独立字节分段，存储(字节序列, 出现次数)
    segments: list[tuple[list[bytes], int]] = []
    for token_str, count in pre_token_counts.items():
        byte_nums = list(token_str.encode("utf-8"))
        seg = [bytes([b]) for b in byte_nums]
        segments.append((seg, count))

    # 统计所有预分段内部相邻字节对频率
    pair_freqs: dict[tuple[bytes, bytes], int] = {}
    for seg, seg_cnt in segments:
        for i in range(len(seg) - 1):
            pair = (seg[i], seg[i+1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + seg_cnt

    merges: list[tuple[bytes, bytes]] = []
    # BPE主循环，持续合并直到词表达到目标大小
    while len(vocab) < vocab_size and pair_freqs:
        # tie-breaking：频次优先，同频次取字典序更大的pair
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
        token_a, token_b = best_pair
        merged_token = token_a + token_b

        # 新增合并子词存入词表，记录merge顺序
        new_id = len(vocab)
        vocab[new_id] = merged_token
        merges.append(best_pair)

        # 遍历所有分段，全局替换本次合并的pair
        new_segments = []
        for seg, seg_cnt in segments:
            new_seg = []
            i = 0
            while i < len(seg):
                if i + 1 < len(seg) and seg[i] == token_a and seg[i+1] == token_b:
                    new_seg.append(merged_token)
                    i += 2
                else:
                    new_seg.append(seg[i])
                    i += 1
            new_segments.append((new_seg, seg_cnt))
        segments = new_segments

        # 清空频率表，下一轮完整重统计所有分段pair
        pair_freqs.clear()
        for seg, seg_cnt in segments:
            for i in range(len(seg) - 1):
                pair = (seg[i], seg[i+1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + seg_cnt

    print(f"BPE 训练完成！最终词表大小: {len(vocab)}, 总合并次数: {len(merges)}")
    return vocab, merges


# 本地测试入口
if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="../tests/fixtures/german.txt",
        vocab_size=300,
        special_tokens=["<s>", "</s>", "<unk>"]
    )