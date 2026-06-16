import math
from typing import List, Tuple

from utils import idx_to_char, BLANK_IDX

def greedy_decode_from_logits(logits) -> List[str]:
    """
    logits: [T, B, C]
    return: list[str]
    """
    preds = logits.argmax(dim=2)
    preds = preds.permute(1, 0)

    results = []
    for seq in preds:
        result = []
        prev = None
        for idx in seq.cpu().tolist():
            if idx != BLANK_IDX and idx != prev:
                result.append(idx_to_char.get(idx, ""))
            prev = idx
        results.append("".join(result))
    return results

def _log_sum_exp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))

def ctc_beam_search_single(log_probs, beam_width: int = 10) -> str:
    """
    对单个样本做 CTC beam search
    log_probs: [T, C]，必须已经是 log_softmax 后的结果
    return: best string
    """
    T, C = log_probs.shape

    beam = {(): (0.0, -math.inf)}

    for t in range(T):
        new_beam = {}

        for prefix, (p_b, p_nb) in beam.items():
            for c in range(C):
                p = float(log_probs[t, c].item())

                if c == BLANK_IDX:

                    n_p_b, n_p_nb = new_beam.get(prefix, (-math.inf, -math.inf))
                    n_p_b = _log_sum_exp(n_p_b, p_b + p)
                    n_p_b = _log_sum_exp(n_p_b, p_nb + p)
                    new_beam[prefix] = (n_p_b, n_p_nb)
                else:
                    end = prefix[-1] if prefix else None
                    new_prefix = prefix + (c,)

                    if c == end:

                        n_p_b, n_p_nb = new_beam.get(new_prefix, (-math.inf, -math.inf))
                        n_p_nb = _log_sum_exp(n_p_nb, p_b + p)
                        new_beam[new_prefix] = (n_p_b, n_p_nb)

                        n_p_b2, n_p_nb2 = new_beam.get(prefix, (-math.inf, -math.inf))
                        n_p_nb2 = _log_sum_exp(n_p_nb2, p_nb + p)
                        new_beam[prefix] = (n_p_b2, n_p_nb2)
                    else:

                        n_p_b, n_p_nb = new_beam.get(new_prefix, (-math.inf, -math.inf))
                        n_p_nb = _log_sum_exp(n_p_nb, p_b + p)
                        n_p_nb = _log_sum_exp(n_p_nb, p_nb + p)
                        new_beam[new_prefix] = (n_p_b, n_p_nb)

        beam_items = []
        for prefix, (p_b, p_nb) in new_beam.items():
            score = _log_sum_exp(p_b, p_nb)
            beam_items.append((prefix, p_b, p_nb, score))

        beam_items.sort(key=lambda x: x[3], reverse=True)
        beam = {prefix: (p_b, p_nb) for prefix, p_b, p_nb, _ in beam_items[:beam_width]}

    best_prefix = max(
        beam.items(),
        key=lambda x: _log_sum_exp(x[1][0], x[1][1])
    )[0]

    return "".join(idx_to_char.get(i, "") for i in best_prefix)

def ctc_beam_search_batch(logits, beam_width: int = 10) -> List[str]:
    """
    logits: [T, B, C]
    return: list[str]
    """
    log_probs = logits.log_softmax(2)
    log_probs = log_probs.permute(1, 0, 2)

    results = []
    for i in range(log_probs.size(0)):
        text = ctc_beam_search_single(log_probs[i], beam_width=beam_width)
        results.append(text)
    return results
