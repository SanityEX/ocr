import csv
import os
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# 1. 路径设置
# ============================================================

ROOT = r"D:\mnist_project\ocr1\recognition\recognition"

GT_TXT = (
    r"D:\mnist_project\ocr1\recognition"
    r"\recognition\gt_recognition.txt"
)

LEVELS = ["10", "20", "30"]

OUTPUT_CSV = (
    r"D:\mnist_project\ocr1"
    r"\parseq_safe_hybrid_recovery_v4_result.csv"
)


# ============================================================
# 2. 模型与图像参数
# ============================================================

IMG_H = 32
IMG_W = 128
MAX_TEXT_LEN = 25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. Safe Recovery 参数
# ============================================================

# 低于该值的位置可以被修改
LOW_CONF_THRESHOLD = 0.88

# 极低置信度位置可以接受普通字符替换
VERY_LOW_CONF_THRESHOLD = 0.55

# 整体置信度达到以下条件时，不修改
SAFE_AVG_CONF_THRESHOLD = 0.95
SAFE_MIN_CONF_THRESHOLD = 0.75

# 最多替换两个字符
MAX_REPLACEMENTS = 2

# 插入和删除均最多一个字符
MAX_INSERTIONS = 1
MAX_DELETIONS = 1

# 候选最低字符重合率
MIN_CHAR_OVERLAP = 0.60

# 最佳候选必须领先第二候选
MIN_SCORE_MARGIN = 0.30

# 最大可接受成本
MAX_REPLACE_COST = 1.80
MAX_INSERT_COST = 1.30
MAX_DELETE_COST = 1.30

# 候选数量
TOP_CANDIDATES = 5

# 错例打印数量
PRINT_WRONG_LIMIT = 30

# True：使用 ISTD-OC 全部 GT 构建词表
# 属于 closed-set evaluation
USE_GT_LEXICON = True

# 可选：开放词表，每行一个单词
EXTERNAL_LEXICON_TXT = None

# 例如：
# EXTERNAL_LEXICON_TXT = (
#     r"D:\mnist_project\ocr1\english_lexicon.txt"
# )


# ============================================================
# 4. 视觉混淆字符
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
# 5. 图像预处理
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
# 6. 候选结构
# ============================================================

@dataclass
class Candidate:
    word: str
    edit_type: str
    score: float
    cost: float
    overlap: float
    changed_positions: List[int]
    description: str


# ============================================================
# 7. 文本与词表工具
# ============================================================

def normalize_text(text: str) -> str:
    """
    转换为大写，仅保留字母和数字。
    """
    return "".join(
        ch for ch in text.upper()
        if ch.isalnum()
    )


def load_gt(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GT 文件不存在：{path}"
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
                    f"[warning] 跳过 GT 第 {line_number} 行："
                    f"{line}"
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
        external_words = load_external_lexicon(
            EXTERNAL_LEXICON_TXT
        )
        words.update(external_words)

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


def is_same_char_type(a: str, b: str) -> bool:
    if a.isalpha() and b.isalpha():
        return True

    if a.isdigit() and b.isdigit():
        return True

    return False


def is_confusion_pair(a: str, b: str) -> bool:
    if a == b:
        return True

    return b in CONFUSION_MAP.get(a, set())


def safe_confidence(
    confidences: Sequence[float],
    index: int,
) -> float:
    if 0 <= index < len(confidences):
        return float(confidences[index])

    return 1.0


def get_low_positions(
    pred: str,
    confidences: Sequence[float],
) -> Set[int]:
    low_positions: Set[int] = set()

    for i in range(len(pred)):
        conf = safe_confidence(
            confidences,
            i,
        )

        if conf < LOW_CONF_THRESHOLD:
            low_positions.add(i)

    return low_positions


# ============================================================
# 8. 绝对安全保护
# ============================================================

def should_keep_raw(
    pred: str,
    confidences: Sequence[float],
    lexicon_set: Set[str],
) -> Tuple[bool, str]:
    """
    重要安全门控。

    1. 预测结果已经在词表中，直接保留。
    2. 平均和最低置信度均较高，直接保留。
    """

    if not pred:
        return True, "empty_prediction"

    # Closed-set 下，正确预测必然在词表中。
    # 因此可以彻底避免把正确结果改错。
    if pred in lexicon_set:
        return True, "raw_in_lexicon"

    valid_conf = list(
        confidences[:len(pred)]
    )

    if not valid_conf:
        return True, "no_confidence"

    avg_conf = sum(valid_conf) / len(valid_conf)
    min_conf = min(valid_conf)

    if (
        avg_conf >= SAFE_AVG_CONF_THRESHOLD
        and min_conf >= SAFE_MIN_CONF_THRESHOLD
    ):
        return True, "high_confidence_raw"

    return False, "allow_recovery"


# ============================================================
# 9. 同长度替换恢复
# ============================================================

def replacement_cost(
    pred_char: str,
    candidate_char: str,
    confidence: float,
) -> float:
    if pred_char == candidate_char:
        return 0.0

    if is_confusion_pair(
        pred_char,
        candidate_char,
    ):
        base = 0.25

    elif is_same_char_type(
        pred_char,
        candidate_char,
    ):
        base = 0.80

    else:
        base = 1.25

    # 置信度越低，修改成本越低
    if confidence < VERY_LOW_CONF_THRESHOLD:
        base *= 0.35

    elif confidence < LOW_CONF_THRESHOLD:
        base *= 0.70

    else:
        # 高置信度位置原则上不会进入修改
        base += 5.0

    return base


def generate_replace_candidates(
    pred: str,
    confidences: Sequence[float],
    words: Sequence[str],
) -> List[Candidate]:
    candidates: List[Candidate] = []

    low_positions = get_low_positions(
        pred,
        confidences,
    )

    if not low_positions:
        return candidates

    for word in words:
        if word == pred:
            continue

        if len(word) != len(pred):
            continue

        changed_positions = [
            i
            for i, (a, b) in enumerate(
                zip(pred, word)
            )
            if a != b
        ]

        if not changed_positions:
            continue

        if len(changed_positions) > MAX_REPLACEMENTS:
            continue

        # 所有变化必须位于低置信度位置
        if any(
            position not in low_positions
            for position in changed_positions
        ):
            continue

        total_cost = 0.0
        valid = True

        for position in changed_positions:
            pred_char = pred[position]
            candidate_char = word[position]

            confidence = safe_confidence(
                confidences,
                position,
            )

            # 中等低置信度位置只能做视觉相似替换
            if (
                confidence >= VERY_LOW_CONF_THRESHOLD
                and not is_confusion_pair(
                    pred_char,
                    candidate_char,
                )
            ):
                valid = False
                break

            total_cost += replacement_cost(
                pred_char,
                candidate_char,
                confidence,
            )

        if not valid:
            continue

        if total_cost > MAX_REPLACE_COST:
            continue

        overlap = char_overlap(
            pred,
            word,
        )

        if overlap < MIN_CHAR_OVERLAP:
            continue

        # 分数越高越好
        score = (
            2.0 * overlap
            - total_cost
            - 0.20 * len(changed_positions)
        )

        description = ",".join(
            (
                f"{position}:"
                f"{pred[position]}"
                f"->{word[position]}"
            )
            for position in changed_positions
        )

        candidates.append(
            Candidate(
                word=word,
                edit_type="replace",
                score=score,
                cost=total_cost,
                overlap=overlap,
                changed_positions=changed_positions,
                description=description,
            )
        )

    return candidates


# ============================================================
# 10. 单字符插入恢复
# ============================================================

def find_single_insertion(
    pred: str,
    candidate: str,
) -> Tuple[bool, int, str]:
    """
    candidate 比 pred 多一个字符。

    返回：
    是否能通过一次插入得到，
    插入位置，
    插入字符。
    """

    if len(candidate) != len(pred) + 1:
        return False, -1, ""

    i = 0
    j = 0
    insertion_position = -1
    inserted_char = ""

    while i < len(pred) and j < len(candidate):
        if pred[i] == candidate[j]:
            i += 1
            j += 1
        else:
            if insertion_position != -1:
                return False, -1, ""

            insertion_position = i
            inserted_char = candidate[j]
            j += 1

    if insertion_position == -1:
        insertion_position = len(pred)
        inserted_char = candidate[-1]

    return True, insertion_position, inserted_char


def insertion_is_near_low_confidence(
    insertion_position: int,
    pred: str,
    confidences: Sequence[float],
) -> bool:
    """
    插入位置必须靠近低置信度字符。

    插在 i 前面时，检查 i-1 和 i。
    """

    nearby_positions = []

    if insertion_position - 1 >= 0:
        nearby_positions.append(
            insertion_position - 1
        )

    if insertion_position < len(pred):
        nearby_positions.append(
            insertion_position
        )

    if not nearby_positions:
        return False

    return any(
        safe_confidence(
            confidences,
            position,
        ) < LOW_CONF_THRESHOLD
        for position in nearby_positions
    )


def generate_insert_candidates(
    pred: str,
    confidences: Sequence[float],
    words: Sequence[str],
) -> List[Candidate]:
    candidates: List[Candidate] = []

    if MAX_INSERTIONS < 1:
        return candidates

    for word in words:
        valid, insertion_position, inserted_char = (
            find_single_insertion(
                pred,
                word,
            )
        )

        if not valid:
            continue

        # 必须在低置信度字符附近插入
        if not insertion_is_near_low_confidence(
            insertion_position,
            pred,
            confidences,
        ):
            continue

        nearby_conf = 1.0

        nearby_positions = []

        if insertion_position - 1 >= 0:
            nearby_positions.append(
                insertion_position - 1
            )

        if insertion_position < len(pred):
            nearby_positions.append(
                insertion_position
            )

        if nearby_positions:
            nearby_conf = min(
                safe_confidence(
                    confidences,
                    position,
                )
                for position in nearby_positions
            )

        if nearby_conf < VERY_LOW_CONF_THRESHOLD:
            cost = 0.35
        else:
            cost = 0.75

        if cost > MAX_INSERT_COST:
            continue

        overlap = char_overlap(
            pred,
            word,
        )

        if overlap < MIN_CHAR_OVERLAP:
            continue

        score = (
            2.0 * overlap
            - cost
            - 0.35
        )

        candidates.append(
            Candidate(
                word=word,
                edit_type="insert",
                score=score,
                cost=cost,
                overlap=overlap,
                changed_positions=[
                    insertion_position
                ],
                description=(
                    f"insert {inserted_char} "
                    f"at {insertion_position}"
                ),
            )
        )

    return candidates


# ============================================================
# 11. 单字符删除恢复
# ============================================================

def find_single_deletion(
    pred: str,
    candidate: str,
) -> Tuple[bool, int, str]:
    """
    pred 比 candidate 多一个字符。

    返回：
    是否能通过删除一个字符得到，
    删除位置，
    删除字符。
    """

    if len(pred) != len(candidate) + 1:
        return False, -1, ""

    i = 0
    j = 0
    deletion_position = -1
    deleted_char = ""

    while i < len(pred) and j < len(candidate):
        if pred[i] == candidate[j]:
            i += 1
            j += 1
        else:
            if deletion_position != -1:
                return False, -1, ""

            deletion_position = i
            deleted_char = pred[i]
            i += 1

    if deletion_position == -1:
        deletion_position = len(pred) - 1
        deleted_char = pred[-1]

    return True, deletion_position, deleted_char


def generate_delete_candidates(
    pred: str,
    confidences: Sequence[float],
    words: Sequence[str],
) -> List[Candidate]:
    candidates: List[Candidate] = []

    if MAX_DELETIONS < 1:
        return candidates

    for word in words:
        valid, deletion_position, deleted_char = (
            find_single_deletion(
                pred,
                word,
            )
        )

        if not valid:
            continue

        confidence = safe_confidence(
            confidences,
            deletion_position,
        )

        # 只能删除低置信度字符
        if confidence >= LOW_CONF_THRESHOLD:
            continue

        if confidence < VERY_LOW_CONF_THRESHOLD:
            cost = 0.30
        else:
            cost = 0.70

        # 词尾低置信度字符更可能是多余输出
        if deletion_position == len(pred) - 1:
            cost -= 0.15

        # 重复字符降低删除成本
        if (
            deletion_position > 0
            and pred[deletion_position]
            == pred[deletion_position - 1]
        ):
            cost -= 0.15

        cost = max(cost, 0.05)

        if cost > MAX_DELETE_COST:
            continue

        overlap = char_overlap(
            pred,
            word,
        )

        if overlap < MIN_CHAR_OVERLAP:
            continue

        score = (
            2.0 * overlap
            - cost
            - 0.25
        )

        candidates.append(
            Candidate(
                word=word,
                edit_type="delete",
                score=score,
                cost=cost,
                overlap=overlap,
                changed_positions=[
                    deletion_position
                ],
                description=(
                    f"delete {deleted_char} "
                    f"at {deletion_position}"
                ),
            )
        )

    return candidates


# ============================================================
# 12. Safe Hybrid Recovery
# ============================================================

def safe_hybrid_recover(
    pred: str,
    confidences: Sequence[float],
    lexicon_set: Set[str],
    length_index: Dict[int, List[str]],
) -> Tuple[
    str,
    List[Candidate],
    str,
]:
    # 第一层：绝对保护
    keep_raw, keep_reason = should_keep_raw(
        pred,
        confidences,
        lexicon_set,
    )

    if keep_raw:
        return pred, [], keep_reason

    all_candidates: List[Candidate] = []

    # A. 同长度替换
    same_length_words = length_index.get(
        len(pred),
        [],
    )

    replace_candidates = (
        generate_replace_candidates(
            pred,
            confidences,
            same_length_words,
        )
    )

    all_candidates.extend(
        replace_candidates
    )

    # B. 仅插入一个字符
    longer_words = length_index.get(
        len(pred) + 1,
        [],
    )

    insert_candidates = (
        generate_insert_candidates(
            pred,
            confidences,
            longer_words,
        )
    )

    all_candidates.extend(
        insert_candidates
    )

    # C. 仅删除一个字符
    shorter_words = length_index.get(
        len(pred) - 1,
        [],
    )

    delete_candidates = (
        generate_delete_candidates(
            pred,
            confidences,
            shorter_words,
        )
    )

    all_candidates.extend(
        delete_candidates
    )

    if not all_candidates:
        return pred, [], "no_safe_candidate"

    all_candidates.sort(
        key=lambda item: (
            -item.score,
            item.cost,
            -item.overlap,
            item.word,
        )
    )

    best = all_candidates[0]

    if len(all_candidates) >= 2:
        second = all_candidates[1]

        margin = (
            best.score
            - second.score
        )
    else:
        margin = 999.0

    # 候选优势不足，不修改
    if margin < MIN_SCORE_MARGIN:
        return (
            pred,
            all_candidates[:TOP_CANDIDATES],
            "insufficient_margin",
        )

    # 最佳候选分数必须为正
    if best.score <= 0:
        return (
            pred,
            all_candidates[:TOP_CANDIDATES],
            "candidate_score_too_low",
        )

    return (
        best.word,
        all_candidates[:TOP_CANDIDATES],
        best.edit_type,
    )


# ============================================================
# 13. PARSeq
# ============================================================

class PARSeqSafeHybridOCR:
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
# 14. 单等级评估
# ============================================================

def evaluate_level(
    model: PARSeqSafeHybridOCR,
    gt_data: Dict[str, str],
    lexicon_set: Set[str],
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

    protected_by_lexicon = 0
    protected_by_confidence = 0

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
            safe_hybrid_recover(
                pred,
                confidences,
                lexicon_set,
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

        if reason == "raw_in_lexicon":
            protected_by_lexicon += 1

        if reason == "high_confidence_raw":
            protected_by_confidence += 1

        if recovered != pred:
            changed += 1

            if not raw_ok and recovered_ok:
                improved += 1

            if raw_ok and not recovered_ok:
                damaged += 1

        elif not recovered_ok:
            unchanged_wrong += 1

        confidence_text = ",".join(
            f"{value:.4f}"
            for value in confidences
        )

        candidate_text = " | ".join(
            (
                f"{candidate.word}:"
                f"{candidate.edit_type}:"
                f"{candidate.score:.4f}:"
                f"{candidate.description}"
            )
            for candidate in candidates
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
            and len(wrong_examples) < PRINT_WRONG_LIMIT
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
                f"safe={recovered_correct / total:.4f} "
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
        "protected_by_lexicon": protected_by_lexicon,
        "protected_by_confidence": protected_by_confidence,
        "wrong_examples": wrong_examples,
        "csv_rows": csv_rows,
    }


# ============================================================
# 15. 打印错例
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
            f"Safe={case['recovered']} | "
            f"avg_conf={case['avg_conf']:.4f} | "
            f"reason={case['reason']}"
        )

        if case["confidences"]:
            confidence_text = " ".join(
                f"{i}:{value:.3f}"
                for i, value in enumerate(
                    case["confidences"]
                )
            )

            print(
                "  confidences:",
                confidence_text,
            )

        if case["candidates"]:
            print("  candidates:")

            for candidate in case["candidates"]:
                print(
                    f"    {candidate.word:20s} "
                    f"type={candidate.edit_type:8s} "
                    f"score={candidate.score:.4f} "
                    f"cost={candidate.cost:.4f} "
                    f"overlap={candidate.overlap:.2f} "
                    f"{candidate.description}"
                )


# ============================================================
# 16. 保存 CSV
# ============================================================

def save_csv(
    rows: Sequence[Dict],
    output_csv: str = OUTPUT_CSV,
) -> None:
    output_dir = os.path.dirname(
        output_csv
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
        output_csv,
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
# 17. 主程序
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PARSeq + Safe Hybrid Recovery V4 on selected ISTD-OC levels."
    )
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--gt-txt", default=GT_TXT)
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--external-lexicon-txt", default=EXTERNAL_LEXICON_TXT)
    parser.add_argument(
        "--open-lexicon",
        action="store_true",
        help="Use only external lexicon instead of GT closed-set lexicon.",
    )
    return parser.parse_args()


def main() -> None:
    global ROOT
    global GT_TXT
    global USE_GT_LEXICON
    global EXTERNAL_LEXICON_TXT

    args = parse_args()
    ROOT = args.root
    GT_TXT = args.gt_txt
    EXTERNAL_LEXICON_TXT = args.external_lexicon_txt

    if args.open_lexicon:
        USE_GT_LEXICON = False

    levels = [
        item.strip()
        for item in args.levels.split(",")
        if item.strip()
    ]

    print("device:", DEVICE)
    print("root:", ROOT)
    print("gt_txt:", GT_TXT)
    print("levels:", levels)
    print("output_csv:", args.output_csv)

    print(
        "low confidence threshold:",
        LOW_CONF_THRESHOLD,
    )

    print(
        "safe average confidence:",
        SAFE_AVG_CONF_THRESHOLD,
    )

    print(
        "safe minimum confidence:",
        SAFE_MIN_CONF_THRESHOLD,
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

    lexicon_set = set(
        lexicon
    )

    length_index = build_length_index(
        lexicon
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

    model = PARSeqSafeHybridOCR(
        DEVICE
    )

    all_csv_rows = []

    print("=" * 76)
    print(
        "PARSeq + Safe Hybrid "
        "Occlusion Recovery V4"
    )
    print("=" * 76)

    for level in levels:
        print("\n" + "=" * 76)
        print("LEVEL:", level)

        result = evaluate_level(
            model,
            gt_data,
            lexicon_set,
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
            "Safe correct:",
            result["recovered_correct"],
        )

        print(
            "PARSeq accuracy:",
            result["raw_acc"],
        )

        print(
            "Safe accuracy:",
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

        print(
            "protected by lexicon:",
            result["protected_by_lexicon"],
        )

        print(
            "protected by confidence:",
            result["protected_by_confidence"],
        )

        print("-" * 76)

        print_wrong_examples(
            result["wrong_examples"]
        )

        all_csv_rows.extend(
            result["csv_rows"]
        )

    save_csv(
        all_csv_rows,
        args.output_csv,
    )

    print("\n" + "=" * 76)
    print("DONE")
    print(
        "saved:",
        args.output_csv,
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
