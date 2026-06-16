import csv
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# 路径设置
# ============================================================

ROOT = r"D:\mnist_project\ocr1\recognition\recognition"

GT_TXT = (
    r"D:\mnist_project\ocr1\recognition"
    r"\recognition\gt_recognition.txt"
)

LEVELS = ["10", "20", "30"]

OUTPUT_CSV = (
    r"D:\mnist_project\ocr1"
    r"\parseq_hybrid_occlusion_recovery_v3_result.csv"
)


# ============================================================
# 模型和图像参数
# ============================================================

IMG_H = 32
IMG_W = 128
MAX_TEXT_LEN = 25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Recovery 参数
# ============================================================

LOW_CONF_THRESHOLD = 0.88
VERY_LOW_CONF_THRESHOLD = 0.55

KEEP_RAW_AVG_CONF = 0.970
KEEP_RAW_MIN_CONF = 0.82

MAX_REPLACEMENTS = 2
MAX_INSERTIONS = 2
MAX_DELETIONS = 2

MAX_TOTAL_EDIT = 3

MIN_SCORE_MARGIN = 0.18
MAX_ACCEPT_COST = 3.20

HIGH_CONF_BREAK_PENALTY = 4.00
MEDIUM_CONF_BREAK_PENALTY = 1.60

REPLACE_CONFUSION_COST = 0.25
REPLACE_NORMAL_COST = 0.85
REPLACE_TYPE_CHANGE_COST = 1.30

INSERT_BASE_COST = 0.72
DELETE_BASE_COST = 0.72

LOW_CONF_EDIT_BONUS = 0.55
VERY_LOW_CONF_EDIT_BONUS = 0.90

CHAR_OVERLAP_WEIGHT = 0.80
ORDER_MATCH_WEIGHT = 1.20
LENGTH_CHANGE_PENALTY = 0.35

USE_GT_LEXICON = True
EXTERNAL_LEXICON_TXT = None

PRINT_WRONG_LIMIT = 30


# ============================================================
# 字符混淆组
# ============================================================

CONFUSION_GROUPS = [
    set("I1LT"),
    set("O0QDCG"),
    set("S5"),
    set("B8R"),
    set("MNWH"),
    set("AEF"),
    set("UVY"),
    set("PFR"),
    set("Z2"),
    set("KX"),
    set("G6C"),
]

CONFUSION_MAP: Dict[str, Set[str]] = {}

for group in CONFUSION_GROUPS:
    for ch in group:
        CONFUSION_MAP.setdefault(ch, set()).update(group)


# ============================================================
# 图像预处理
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# ============================================================
# 数据结构
# ============================================================

@dataclass
class AlignmentResult:
    cost: float
    replacements: int
    insertions: int
    deletions: int
    high_conf_breaks: int
    low_conf_edits: int
    operations: List[Tuple]


@dataclass
class CandidateResult:
    word: str
    final_score: float
    alignment_cost: float
    replacements: int
    insertions: int
    deletions: int
    overlap: float
    order_score: float
    high_conf_breaks: int
    low_conf_edits: int
    operations: List[Tuple]


# ============================================================
# 文本工具
# ============================================================

def normalize_text(text: str) -> str:
    return "".join(
        ch for ch in text.upper()
        if ch.isalnum()
    )


def load_gt(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GT文件不存在：{path}"
        )

    data: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 2:
                print(
                    f"[warning] 跳过第{line_number}行：{line}"
                )
                continue

            filename = parts[0].strip()
            label = ",".join(parts[1:]).strip()
            label = label.replace('"', "")
            label = normalize_text(label)

            if label:
                data[filename] = label

    return data


def load_external_lexicon(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"外部词表不存在：{path}"
        )

    words: Set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = normalize_text(line.strip())

            if 2 <= len(word) <= MAX_TEXT_LEN:
                words.add(word)

    return sorted(words)


def build_lexicon(
    gt_data: Dict[str, str],
) -> List[str]:
    words: Set[str] = set()

    if USE_GT_LEXICON:
        for label in gt_data.values():
            word = normalize_text(label)

            if 2 <= len(word) <= MAX_TEXT_LEN:
                words.add(word)

    if EXTERNAL_LEXICON_TXT:
        words.update(
            load_external_lexicon(
                EXTERNAL_LEXICON_TXT
            )
        )

    return sorted(words)


def build_length_index(
    lexicon: Sequence[str],
) -> Dict[int, List[str]]:
    index: Dict[int, List[str]] = {}

    for word in lexicon:
        index.setdefault(len(word), []).append(word)

    return index


def char_overlap(a: str, b: str) -> float:
    ca = Counter(a)
    cb = Counter(b)

    common = sum(
        min(ca[ch], cb.get(ch, 0))
        for ch in ca
    )

    return common / max(len(a), len(b), 1)


def lcs_length(a: str, b: str) -> int:
    n = len(a)
    m = len(b)

    previous = [0] * (m + 1)

    for i in range(1, n + 1):
        current = [0] * (m + 1)

        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(
                    previous[j],
                    current[j - 1],
                )

        previous = current

    return previous[m]


def order_match_score(a: str, b: str) -> float:
    return lcs_length(a, b) / max(len(a), len(b), 1)


def is_confusion_pair(a: str, b: str) -> bool:
    if a == b:
        return True

    return b in CONFUSION_MAP.get(a, set())


def is_same_char_type(a: str, b: str) -> bool:
    if a.isalpha() and b.isalpha():
        return True

    if a.isdigit() and b.isdigit():
        return True

    return False


def safe_confidence(
    confidences: Sequence[float],
    index: int,
) -> float:
    if 0 <= index < len(confidences):
        return float(confidences[index])

    return 0.50


def get_low_positions(
    pred: str,
    confidences: Sequence[float],
) -> Set[int]:
    positions: Set[int] = set()

    for i in range(len(pred)):
        conf = safe_confidence(
            confidences,
            i,
        )

        if conf < LOW_CONF_THRESHOLD:
            positions.add(i)

    return positions


# ============================================================
# 编辑成本
# ============================================================

def replacement_cost(
    pred_char: str,
    candidate_char: str,
    confidence: float,
) -> Tuple[float, bool, bool]:
    if pred_char == candidate_char:
        return 0.0, False, False

    if is_confusion_pair(pred_char, candidate_char):
        base = REPLACE_CONFUSION_COST

    elif is_same_char_type(pred_char, candidate_char):
        base = REPLACE_NORMAL_COST

    else:
        base = REPLACE_TYPE_CHANGE_COST

    high_break = False
    low_edit = False

    if confidence >= LOW_CONF_THRESHOLD:
        base += HIGH_CONF_BREAK_PENALTY
        high_break = True

    elif confidence >= VERY_LOW_CONF_THRESHOLD:
        base += MEDIUM_CONF_BREAK_PENALTY

    else:
        base -= VERY_LOW_CONF_EDIT_BONUS
        low_edit = True

    if (
        confidence < LOW_CONF_THRESHOLD
        and confidence >= VERY_LOW_CONF_THRESHOLD
    ):
        base -= LOW_CONF_EDIT_BONUS
        low_edit = True

    return max(base, 0.02), high_break, low_edit


def insertion_cost(
    pred: str,
    candidate_index: int,
    pred_index: int,
    confidences: Sequence[float],
) -> Tuple[float, bool]:
    nearby_confidences = []

    if pred_index - 1 >= 0:
        nearby_confidences.append(
            safe_confidence(
                confidences,
                pred_index - 1,
            )
        )

    if pred_index < len(pred):
        nearby_confidences.append(
            safe_confidence(
                confidences,
                pred_index,
            )
        )

    nearby_conf = (
        min(nearby_confidences)
        if nearby_confidences
        else 0.50
    )

    cost = INSERT_BASE_COST
    low_edit = False

    if nearby_conf < VERY_LOW_CONF_THRESHOLD:
        cost -= VERY_LOW_CONF_EDIT_BONUS
        low_edit = True

    elif nearby_conf < LOW_CONF_THRESHOLD:
        cost -= LOW_CONF_EDIT_BONUS
        low_edit = True

    else:
        cost += MEDIUM_CONF_BREAK_PENALTY

    return max(cost, 0.05), low_edit


def deletion_cost(
    pred: str,
    pred_index: int,
    confidences: Sequence[float],
) -> Tuple[float, bool, bool]:
    confidence = safe_confidence(
        confidences,
        pred_index,
    )

    cost = DELETE_BASE_COST
    high_break = False
    low_edit = False

    if confidence >= LOW_CONF_THRESHOLD:
        cost += HIGH_CONF_BREAK_PENALTY
        high_break = True

    elif confidence >= VERY_LOW_CONF_THRESHOLD:
        cost += MEDIUM_CONF_BREAK_PENALTY
        cost -= LOW_CONF_EDIT_BONUS
        low_edit = True

    else:
        cost -= VERY_LOW_CONF_EDIT_BONUS
        low_edit = True

    # 词尾低置信度字符更可能是多余输出
    if pred_index == len(pred) - 1:
        cost -= 0.20

    # 重复字符删除成本降低
    if (
        pred_index > 0
        and pred[pred_index] == pred[pred_index - 1]
    ):
        cost -= 0.25

    return max(cost, 0.05), high_break, low_edit


# ============================================================
# 动态规划对齐
# ============================================================

def align_candidate(
    pred: str,
    candidate: str,
    confidences: Sequence[float],
) -> AlignmentResult:
    n = len(pred)
    m = len(candidate)

    inf = float("inf")

    dp = [
        [
            {
                "cost": inf,
                "replacements": 0,
                "insertions": 0,
                "deletions": 0,
                "high_conf_breaks": 0,
                "low_conf_edits": 0,
                "operations": [],
            }
            for _ in range(m + 1)
        ]
        for _ in range(n + 1)
    ]

    dp[0][0]["cost"] = 0.0

    def update(
        ni: int,
        nj: int,
        current: Dict,
        add_cost: float,
        replacements: int,
        insertions: int,
        deletions: int,
        high_breaks: int,
        low_edits: int,
        operation: Tuple,
    ):
        new_cost = current["cost"] + add_cost

        if new_cost < dp[ni][nj]["cost"]:
            dp[ni][nj] = {
                "cost": new_cost,
                "replacements": (
                    current["replacements"]
                    + replacements
                ),
                "insertions": (
                    current["insertions"]
                    + insertions
                ),
                "deletions": (
                    current["deletions"]
                    + deletions
                ),
                "high_conf_breaks": (
                    current["high_conf_breaks"]
                    + high_breaks
                ),
                "low_conf_edits": (
                    current["low_conf_edits"]
                    + low_edits
                ),
                "operations": (
                    current["operations"]
                    + [operation]
                ),
            }

    for i in range(n + 1):
        for j in range(m + 1):
            current = dp[i][j]

            if math.isinf(current["cost"]):
                continue

            # 相同或替换
            if i < n and j < m:
                pred_char = pred[i]
                candidate_char = candidate[j]

                if pred_char == candidate_char:
                    update(
                        i + 1,
                        j + 1,
                        current,
                        0.0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        (
                            "match",
                            i,
                            j,
                            pred_char,
                            candidate_char,
                        ),
                    )

                else:
                    conf = safe_confidence(
                        confidences,
                        i,
                    )

                    cost, high_break, low_edit = (
                        replacement_cost(
                            pred_char,
                            candidate_char,
                            conf,
                        )
                    )

                    update(
                        i + 1,
                        j + 1,
                        current,
                        cost,
                        1,
                        0,
                        0,
                        int(high_break),
                        int(low_edit),
                        (
                            "replace",
                            i,
                            j,
                            pred_char,
                            candidate_char,
                            conf,
                        ),
                    )

            # candidate多一个字符：插入
            if j < m:
                cost, low_edit = insertion_cost(
                    pred,
                    j,
                    i,
                    confidences,
                )

                update(
                    i,
                    j + 1,
                    current,
                    cost,
                    0,
                    1,
                    0,
                    0,
                    int(low_edit),
                    (
                        "insert",
                        i,
                        j,
                        "",
                        candidate[j],
                    ),
                )

            # pred多一个字符：删除
            if i < n:
                cost, high_break, low_edit = (
                    deletion_cost(
                        pred,
                        i,
                        confidences,
                    )
                )

                update(
                    i + 1,
                    j,
                    current,
                    cost,
                    0,
                    0,
                    1,
                    int(high_break),
                    int(low_edit),
                    (
                        "delete",
                        i,
                        j,
                        pred[i],
                        "",
                        safe_confidence(
                            confidences,
                            i,
                        ),
                    ),
                )

    result = dp[n][m]

    return AlignmentResult(
        cost=float(result["cost"]),
        replacements=int(result["replacements"]),
        insertions=int(result["insertions"]),
        deletions=int(result["deletions"]),
        high_conf_breaks=int(
            result["high_conf_breaks"]
        ),
        low_conf_edits=int(
            result["low_conf_edits"]
        ),
        operations=list(result["operations"]),
    )


# ============================================================
# 候选过滤
# ============================================================

def candidate_length_allowed(
    pred: str,
    candidate: str,
) -> bool:
    length_diff = len(candidate) - len(pred)

    if length_diff > MAX_INSERTIONS:
        return False

    if length_diff < -MAX_DELETIONS:
        return False

    return True


def candidate_alignment_allowed(
    alignment: AlignmentResult,
) -> bool:
    if alignment.replacements > MAX_REPLACEMENTS:
        return False

    if alignment.insertions > MAX_INSERTIONS:
        return False

    if alignment.deletions > MAX_DELETIONS:
        return False

    total_edit = (
        alignment.replacements
        + alignment.insertions
        + alignment.deletions
    )

    if total_edit > MAX_TOTAL_EDIT:
        return False

    # 自动恢复阶段不允许破坏多个高置信度字符
    if alignment.high_conf_breaks > 0:
        return False

    return True


def calculate_candidate_score(
    pred: str,
    candidate: str,
    alignment: AlignmentResult,
) -> CandidateResult:
    overlap = char_overlap(
        candidate,
        pred,
    )

    order_score = order_match_score(
        candidate,
        pred,
    )

    length_diff = abs(
        len(candidate) - len(pred)
    )

    final_score = (
        -alignment.cost
        + CHAR_OVERLAP_WEIGHT * overlap
        + ORDER_MATCH_WEIGHT * order_score
        + LOW_CONF_EDIT_BONUS
        * alignment.low_conf_edits
        - LENGTH_CHANGE_PENALTY
        * length_diff
    )

    return CandidateResult(
        word=candidate,
        final_score=final_score,
        alignment_cost=alignment.cost,
        replacements=alignment.replacements,
        insertions=alignment.insertions,
        deletions=alignment.deletions,
        overlap=overlap,
        order_score=order_score,
        high_conf_breaks=alignment.high_conf_breaks,
        low_conf_edits=alignment.low_conf_edits,
        operations=alignment.operations,
    )


# ============================================================
# Hybrid Recovery
# ============================================================

def hybrid_recover(
    pred: str,
    confidences: Sequence[float],
    length_index: Dict[int, List[str]],
) -> Tuple[
    str,
    List[CandidateResult],
    str,
]:
    if not pred:
        return pred, [], "empty_prediction"

    valid_confidences = list(
        confidences[:len(pred)]
    )

    avg_conf = (
        sum(valid_confidences)
        / len(valid_confidences)
        if valid_confidences
        else 0.0
    )

    min_conf = min(
        valid_confidences,
        default=0.0,
    )

    candidate_words: List[str] = []

    min_len = max(
        2,
        len(pred) - MAX_DELETIONS,
    )

    max_len = min(
        MAX_TEXT_LEN,
        len(pred) + MAX_INSERTIONS,
    )

    for length in range(
        min_len,
        max_len + 1,
    ):
        candidate_words.extend(
            length_index.get(
                length,
                [],
            )
        )

    if not candidate_words:
        return pred, [], "no_candidate_length"

    candidates: List[CandidateResult] = []

    for word in candidate_words:
        if word == pred:
            continue

        if not candidate_length_allowed(
            pred,
            word,
        ):
            continue

        alignment = align_candidate(
            pred,
            word,
            confidences,
        )

        if not candidate_alignment_allowed(
            alignment
        ):
            continue

        result = calculate_candidate_score(
            pred,
            word,
            alignment,
        )

        if result.alignment_cost > MAX_ACCEPT_COST:
            continue

        candidates.append(result)

    if not candidates:
        return pred, [], "no_valid_candidate"

    candidates.sort(
        key=lambda item: (
            -item.final_score,
            item.alignment_cost,
            item.replacements
            + item.insertions
            + item.deletions,
            -item.overlap,
            item.word,
        )
    )

    best = candidates[0]

    if len(candidates) >= 2:
        second = candidates[1]
        margin = (
            best.final_score
            - second.final_score
        )
    else:
        margin = 999.0

    # 原结果置信度高时，需要更强的候选优势
    required_margin = MIN_SCORE_MARGIN

    if (
        avg_conf >= KEEP_RAW_AVG_CONF
        and min_conf >= KEEP_RAW_MIN_CONF
    ):
        required_margin += 0.35

    if margin < required_margin:
        return (
            pred,
            candidates[:5],
            "insufficient_margin",
        )

    # 只靠一个低代价普通替换，也可能把正确词改错
    if (
        avg_conf >= KEEP_RAW_AVG_CONF
        and best.replacements > 0
        and best.insertions == 0
        and best.deletions == 0
        and best.low_conf_edits == 0
    ):
        return (
            pred,
            candidates[:5],
            "protect_high_conf_raw",
        )

    edit_type = []

    if best.replacements:
        edit_type.append(
            f"replace:{best.replacements}"
        )

    if best.insertions:
        edit_type.append(
            f"insert:{best.insertions}"
        )

    if best.deletions:
        edit_type.append(
            f"delete:{best.deletions}"
        )

    reason = "+".join(edit_type)

    return (
        best.word,
        candidates[:5],
        reason,
    )


# ============================================================
# PARSeq
# ============================================================

class PARSeqHybridOCR:
    def __init__(
        self,
        device: torch.device,
    ):
        self.device = device

        print("loading PARSeq...")

        self.model = torch.hub.load(
            "baudm/parseq",
            "parseq",
            pretrained=True,
            trust_repo=True,
        ).to(device)

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
    ) -> Tuple[str, List[float], float]:
        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as exc:
            raise RuntimeError(
                f"图片读取失败：{image_path}"
            ) from exc

        x = transform(image)
        x = x.unsqueeze(0).to(self.device)

        logits = self.model(x)
        probabilities = logits.softmax(-1)

        labels, confidence_tensor = (
            self.model.tokenizer.decode(
                probabilities
            )
        )

        pred = normalize_text(
            labels[0]
        )

        confidences: List[float] = []

        if len(confidence_tensor[0]) > 0:
            for confidence in confidence_tensor[0]:
                confidences.append(
                    float(confidence.item())
                )

        confidences = confidences[:len(pred)]

        avg_conf = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0.0
        )

        return pred, confidences, avg_conf


# ============================================================
# 评估
# ============================================================

def evaluate_level(
    model: PARSeqHybridOCR,
    gt_data: Dict[str, str],
    length_index: Dict[int, List[str]],
    level: str,
) -> Dict:
    image_dir = os.path.join(
        ROOT,
        level,
    )

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"图片目录不存在：{image_dir}"
        )

    total = 0
    raw_correct = 0
    recovered_correct = 0

    changed = 0
    improved = 0
    damaged = 0
    unchanged_wrong = 0

    wrong_examples = []
    csv_rows = []

    for filename in sorted(
        os.listdir(image_dir)
    ):
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        gt_name = filename.replace(
            "img_",
            "word_",
        )

        gt_name = (
            os.path.splitext(gt_name)[0]
            + ".png"
        )

        if gt_name not in gt_data:
            continue

        gt = normalize_text(
            gt_data[gt_name]
        )

        if not gt:
            continue

        image_path = os.path.join(
            image_dir,
            filename,
        )

        try:
            pred, confidences, avg_conf = (
                model.predict(image_path)
            )

        except Exception as exc:
            print(
                f"[warning] 跳过 {filename}: {exc}"
            )
            continue

        recovered, candidates, reason = (
            hybrid_recover(
                pred,
                confidences,
                length_index,
            )
        )

        total += 1

        raw_ok = pred == gt
        recovered_ok = recovered == gt

        if raw_ok:
            raw_correct += 1

        if recovered_ok:
            recovered_correct += 1

        if recovered != pred:
            changed += 1

            if not raw_ok and recovered_ok:
                improved += 1

            if raw_ok and not recovered_ok:
                damaged += 1

        elif not recovered_ok:
            unchanged_wrong += 1

        candidate_text = " | ".join(
            (
                f"{item.word}:"
                f"{item.final_score:.4f}:"
                f"r{item.replacements}:"
                f"i{item.insertions}:"
                f"d{item.deletions}"
            )
            for item in candidates
        )

        confidence_text = ",".join(
            f"{value:.4f}"
            for value in confidences
        )

        csv_rows.append({
            "level": level,
            "filename": filename,
            "gt": gt,
            "raw_prediction": pred,
            "recovered_prediction": recovered,
            "raw_correct": int(raw_ok),
            "recovered_correct": int(recovered_ok),
            "average_confidence": avg_conf,
            "confidence_list": confidence_text,
            "recovery_reason": reason,
            "candidates": candidate_text,
        })

        if (
            not recovered_ok
            and len(wrong_examples)
            < PRINT_WRONG_LIMIT
        ):
            wrong_examples.append({
                "filename": filename,
                "gt": gt,
                "pred": pred,
                "recovered": recovered,
                "avg_conf": avg_conf,
                "confidences": confidences,
                "reason": reason,
                "candidates": candidates,
            })

        if total % 200 == 0:
            print(
                f"[level {level}] [{total}] "
                f"raw={raw_correct / total:.4f} "
                f"hybrid={recovered_correct / total:.4f} "
                f"changed={changed} "
                f"improved={improved} "
                f"damaged={damaged}"
            )

    raw_acc = (
        raw_correct / total
        if total > 0
        else 0.0
    )

    recovered_acc = (
        recovered_correct / total
        if total > 0
        else 0.0
    )

    return {
        "level": level,
        "total": total,
        "raw_correct": raw_correct,
        "recovered_correct": recovered_correct,
        "raw_acc": raw_acc,
        "recovered_acc": recovered_acc,
        "gain": recovered_acc - raw_acc,
        "changed": changed,
        "improved": improved,
        "damaged": damaged,
        "unchanged_wrong": unchanged_wrong,
        "wrong_examples": wrong_examples,
        "csv_rows": csv_rows,
    }


# ============================================================
# 输出
# ============================================================

def print_wrong_examples(
    wrong_examples: Sequence[Dict],
) -> None:
    print("\nWrong examples:\n")

    for case in wrong_examples:
        print(
            f"{case['filename']} | "
            f"GT={case['gt']} | "
            f"PARSeq={case['pred']} | "
            f"Hybrid={case['recovered']} | "
            f"avg_conf={case['avg_conf']:.4f} | "
            f"reason={case['reason']}"
        )

        if case["confidences"]:
            print(
                "  confidences:",
                " ".join(
                    f"{i}:{value:.3f}"
                    for i, value in enumerate(
                        case["confidences"]
                    )
                )
            )

        if case["candidates"]:
            print("  candidates:")

            for item in case["candidates"]:
                print(
                    f"    {item.word:20s} "
                    f"final={item.final_score:.4f} "
                    f"cost={item.alignment_cost:.4f} "
                    f"replace={item.replacements} "
                    f"insert={item.insertions} "
                    f"delete={item.deletions} "
                    f"overlap={item.overlap:.2f} "
                    f"order={item.order_score:.2f} "
                    f"low_edits={item.low_conf_edits}"
                )

                operation_text = []

                for operation in item.operations:
                    if operation[0] == "match":
                        continue

                    operation_text.append(
                        str(operation)
                    )

                if operation_text:
                    print(
                        "      operations:",
                        " | ".join(
                            operation_text
                        )
                    )


def save_csv(
    rows: Sequence[Dict],
) -> None:
    output_dir = os.path.dirname(
        OUTPUT_CSV
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    fieldnames = [
        "level",
        "filename",
        "gt",
        "raw_prediction",
        "recovered_prediction",
        "raw_correct",
        "recovered_correct",
        "average_confidence",
        "confidence_list",
        "recovery_reason",
        "candidates",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    print("device:", DEVICE)
    print(
        "low confidence threshold:",
        LOW_CONF_THRESHOLD,
    )
    print(
        "very low confidence threshold:",
        VERY_LOW_CONF_THRESHOLD,
    )
    print(
        "max replacements:",
        MAX_REPLACEMENTS,
    )
    print(
        "max insertions:",
        MAX_INSERTIONS,
    )
    print(
        "max deletions:",
        MAX_DELETIONS,
    )

    gt_data = load_gt(
        GT_TXT
    )

    print(
        "GT samples:",
        len(gt_data),
    )

    lexicon = build_lexicon(
        gt_data
    )

    if not lexicon:
        raise RuntimeError(
            "候选词表为空。"
        )

    print(
        "lexicon size:",
        len(lexicon),
    )

    if USE_GT_LEXICON:
        print(
            "evaluation mode: "
            "closed-set GT lexicon"
        )
    else:
        print(
            "evaluation mode: "
            "external/open lexicon"
        )

    length_index = build_length_index(
        lexicon
    )

    model = PARSeqHybridOCR(
        DEVICE
    )

    all_csv_rows = []

    print("=" * 76)
    print(
        "PARSeq + Hybrid Occlusion "
        "Recovery V3"
    )
    print("=" * 76)

    for level in LEVELS:
        print("\n" + "=" * 76)
        print("LEVEL:", level)

        result = evaluate_level(
            model,
            gt_data,
            length_index,
            level,
        )

        print("-" * 76)
        print(
            "total:",
            result["total"],
        )
        print(
            "PARSeq correct:",
            result["raw_correct"],
        )
        print(
            "Hybrid correct:",
            result["recovered_correct"],
        )
        print(
            "PARSeq accuracy:",
            result["raw_acc"],
        )
        print(
            "Hybrid accuracy:",
            result["recovered_acc"],
        )
        print(
            "gain:",
            result["gain"],
        )
        print(
            "changed:",
            result["changed"],
        )
        print(
            "improved:",
            result["improved"],
        )
        print(
            "damaged:",
            result["damaged"],
        )
        print(
            "unchanged wrong:",
            result["unchanged_wrong"],
        )
        print("-" * 76)

        print_wrong_examples(
            result["wrong_examples"]
        )

        all_csv_rows.extend(
            result["csv_rows"]
        )

    save_csv(
        all_csv_rows
    )

    print("\n" + "=" * 76)
    print("DONE")
    print(
        "saved:",
        OUTPUT_CSV,
    )
    print("=" * 76)


if __name__ == "__main__":
    main()