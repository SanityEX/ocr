CHARS = "abcdefghijklmnopqrstuvwxyz"
BLANK_IDX = 0

char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}


def encode_text(text: str) -> list[int]:
    text = text.lower().strip()
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode_ctc(indices: list[int]) -> str:
    result = []
    prev = None

    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            result.append(idx_to_char.get(idx, ""))
        prev = idx

    return "".join(result)