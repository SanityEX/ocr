CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

# 普通字符从 3 开始
char_to_idx = {c: i + 3 for i, c in enumerate(CHARS)}
idx_to_char = {i + 3: c for i, c in enumerate(CHARS)}

VOCAB_SIZE = len(CHARS) + 3


def encode_text_ctc(text: str) -> list[int]:
    text = text.strip()
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode_ctc(indices: list[int]) -> str:
    result = []
    prev = None

    for idx in indices:
        if idx >= 3 and idx != prev:
            result.append(idx_to_char.get(idx, ""))
        prev = idx

    return "".join(result)


def encode_text_attention(text: str, max_len: int | None = None) -> list[int]:
    """
    输出: [SOS] + chars + [EOS]
    """
    text = text.strip()
    seq = [SOS_IDX]
    seq.extend([char_to_idx[c] for c in text if c in char_to_idx])
    seq.append(EOS_IDX)

    if max_len is not None:
        seq = seq[:max_len]
    return seq


def decode_attention(indices: list[int]) -> str:
    chars = []
    for idx in indices:
        if idx == EOS_IDX:
            break
        if idx >= 3:
            chars.append(idx_to_char.get(idx, ""))
    return "".join(chars)