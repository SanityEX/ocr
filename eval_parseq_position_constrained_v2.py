import csv
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from PIL import Image
from torchvision import transforms

ROOT = r"D:\mnist_project\ocr1\recognition\recognition"

GT_TXT = (
    r"D:\mnist_project\ocr1\recognition"
    r"\recognition\gt_recognition.txt"
)

LEVELS = ["10", "20", "30"]

OUTPUT_CSV = (
    r"D:\mnist_project\ocr1"
    r"\parseq_position_constrained_v2_result.csv"
)

IMG_H = 32
IMG_W = 128

MAX_TEXT_LEN = 25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LOW_CONF_THRESHOLD = 0.88

VERY_LOW_CONF_THRESHOLD = 0.58

HIGH_AVG_CONF_THRESHOLD = 0.965

MAX_REPLACE_COUNT = 3

MIN_CANDIDATE_MARGIN = 0.12

MAX_ACCEPT_COST = 1.65

MIN_CHAR_OVERLAP = 0.65

USE_GT_LEXICON = True

EXTERNAL_LEXICON_TXT = None

CONFUSION_GROUPS = [
    set("I1L"),
    set("O0QDCG"),
    set("S5"),
    set("B8R"),
    set("MNWH"),
    set("AEF"),
    set("UVY"),
    set("PFR"),
    set("T7I"),
    set("Z2"),
    set("G6C"),
    set("KX"),
]

CONFUSION_MAP: Dict[str, Set[str]] = {}

for group in CONFUSION_GROUPS:
    for ch in group:
        CONFUSION_MAP.setdefault(ch, set()).update(group)

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

def normalize_text(text: str) -> str:
    """
    只保留英文数字，并转换为大写。
    """
    return "".join(
        ch for ch in text.upper()
        if ch.isalnum()
    )

def load_gt(path: str) -> Dict[str, str]:
    """
    读取 ISTD-OC GT。
    格式一般为：
    word_1.png,"TEXT"
    """
    data: Dict[str, str] = {}

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GT文件不存在：{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 2:
                print(
                    f"[warning] 跳过GT第{line_number}行："
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

def load_external_lexicon(
    path: str,
) -> List[str]:
    words: Set[str] = set()

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"外部词表不存在：{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = normalize_text(line.strip())

            if 2 <= len(word) <= MAX_TEXT_LEN:
                words.add(word)

    return sorted(words)

def build_lexicon(
    gt_data: Dict[str, str],
) -> List[str]:
    """
    构建候选词表。

    USE_GT_LEXICON=True：
        使用GT中出现过的词，属于闭集评估。

    EXTERNAL_LEXICON_TXT不为空：
        加入外部词表。
    """
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
    """
    按长度索引词表，减少搜索时间。
    """
    index: Dict[int, List[str]] = {}

    for word in lexicon:
        index.setdefault(len(word), []).append(word)

    return index

def char_overlap(a: str, b: str) -> float:
    """
    计算字符集合重合度。
    """
    ca = Counter(a)
    cb = Counter(b)

    common = sum(
        min(ca[ch], cb.get(ch, 0))
        for ch in ca
    )

    return common / max(len(a), len(b), 1)

def is_same_char_type(a: str, b: str) -> bool:
    """
    判断两个字符是否同为数字或同为字母。
    """
    if a.isdigit() and b.isdigit():
        return True

    if a.isalpha() and b.isalpha():
        return True

    return False

def is_confusion_pair(a: str, b: str) -> bool:
    """
    判断两个字符是否属于视觉相似字符。
    """
    if a == b:
        return True

    return b in CONFUSION_MAP.get(a, set())

def get_low_conf_positions(
    pred: str,
    confidences: Sequence[float],
) -> Set[int]:
    """
    返回允许修改的位置。
    """
    low_positions: Set[int] = set()

    usable_length = min(
        len(pred),
        len(confidences),
    )

    for i in range(usable_length):
        if confidences[i] < LOW_CONF_THRESHOLD:
            low_positions.add(i)

    return low_positions

def count_differences(
    word: str,
    pred: str,
) -> int:
    return sum(
        a != b
        for a, b in zip(word, pred)
    )

def candidate_is_valid(
    word: str,
    pred: str,
    confidences: Sequence[float],
    low_positions: Set[int],
) -> bool:
    """
    严格候选过滤：

    1. 必须与预测结果同长度
    2. 高置信度字符完全固定
    3. 修改位置必须属于低置信度位置
    4. 修改数量不能太多
    5. 中等低置信度位置只允许相似字符替换
    6. 字母/数字类型尽量保持一致
    """
    if len(word) != len(pred):
        return False

    replace_count = 0

    for i, (pred_ch, word_ch) in enumerate(
        zip(pred, word)
    ):
        if pred_ch == word_ch:
            continue

        replace_count += 1

        if replace_count > MAX_REPLACE_COUNT:
            return False

        if i not in low_positions:
            return False

        confidence = (
            confidences[i]
            if i < len(confidences)
            else 1.0
        )

        if not is_same_char_type(
            pred_ch,
            word_ch,
        ):
            if confidence >= VERY_LOW_CONF_THRESHOLD:
                return False

        if confidence >= VERY_LOW_CONF_THRESHOLD:
            if not is_confusion_pair(
                pred_ch,
                word_ch,
            ):
                return False

    return replace_count > 0

def substitution_cost(
    pred_ch: str,
    word_ch: str,
    confidence: float,
) -> float:
    """
    置信度越高，修改成本越高。
    视觉相似字符成本更低。
    """
    if pred_ch == word_ch:
        return 0.0

    if is_confusion_pair(pred_ch, word_ch):
        base_cost = 0.32
    elif is_same_char_type(pred_ch, word_ch):
        base_cost = 0.72
    else:
        base_cost = 1.15

    confidence_factor = 0.45 + confidence

    return base_cost * confidence_factor

def candidate_cost(
    word: str,
    pred: str,
    confidences: Sequence[float],
) -> float:
    """
    分数越低越好。
    """
    total_cost = 0.0

    for i, (pred_ch, word_ch) in enumerate(
        zip(pred, word)
    ):
        confidence = (
            confidences[i]
            if i < len(confidences)
            else 1.0
        )

        total_cost += substitution_cost(
            pred_ch,
            word_ch,
            confidence,
        )

    overlap = char_overlap(word, pred)

    overlap_bonus = overlap * 0.20

    total_cost -= overlap_bonus

    return total_cost

def position_constrained_recover(
    pred: str,
    confidences: Sequence[float],
    length_index: Dict[int, List[str]],
) -> Tuple[
    str,
    List[Tuple[str, float, int, float]],
    Set[int],
]:
    """
    执行同长度、位置约束恢复。
    """
    if not pred:
        return pred, [], set()

    if len(pred) not in length_index:
        return pred, [], set()

    low_positions = get_low_conf_positions(
        pred,
        confidences,
    )

    if not low_positions:
        return pred, [], low_positions

    avg_conf = (
        sum(confidences[:len(pred)]) /
        max(len(confidences[:len(pred)]), 1)
    )

    min_conf = min(
        confidences[:len(pred)],
        default=1.0,
    )

    if (
        avg_conf >= HIGH_AVG_CONF_THRESHOLD
        and min_conf >= VERY_LOW_CONF_THRESHOLD
    ):
        return pred, [], low_positions

    candidates: List[
        Tuple[str, float, int, float]
    ] = []

    for word in length_index[len(pred)]:
        if word == pred:
            continue

        if not candidate_is_valid(
            word,
            pred,
            confidences,
            low_positions,
        ):
            continue

        overlap = char_overlap(
            word,
            pred,
        )

        if overlap < MIN_CHAR_OVERLAP:
            continue

        replace_count = count_differences(
            word,
            pred,
        )

        cost = candidate_cost(
            word,
            pred,
            confidences,
        )

        candidates.append(
            (
                word,
                cost,
                replace_count,
                overlap,
            )
        )

    if not candidates:
        return pred, [], low_positions

    candidates.sort(
        key=lambda item: (
            item[1],
            item[2],
            -item[3],
            item[0],
        )
    )

    best_word = candidates[0][0]
    best_cost = candidates[0][1]

    if len(candidates) >= 2:
        second_cost = candidates[1][1]
        margin = second_cost - best_cost
    else:
        margin = 999.0

    if best_cost > MAX_ACCEPT_COST:
        return pred, candidates[:5], low_positions

    if margin < MIN_CANDIDATE_MARGIN:
        return pred, candidates[:5], low_positions

    return (
        best_word,
        candidates[:5],
        low_positions,
    )

class PARSeqPositionConstrainedOCR:
    def __init__(self, device: torch.device):
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

        labels, confidences = (
            self.model.tokenizer.decode(
                probabilities
            )
        )

        pred = normalize_text(labels[0])

        confidence_list: List[float] = []

        if len(confidences[0]) > 0:
            for confidence in confidences[0]:
                confidence_list.append(
                    float(confidence.item())
                )

        confidence_list = (
            confidence_list[:len(pred)]
        )

        avg_conf = (
            sum(confidence_list) /
            len(confidence_list)
            if confidence_list
            else 0.0
        )

        return (
            pred,
            confidence_list,
            avg_conf,
        )

def evaluate_level(
    model: PARSeqPositionConstrainedOCR,
    gt_data: Dict[str, str],
    length_index: Dict[int, List[str]],
    level: str,
) -> Dict:
    img_dir = os.path.join(
        ROOT,
        level,
    )

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"图片目录不存在：{img_dir}"
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

    filenames = sorted(
        os.listdir(img_dir)
    )

    for filename in filenames:
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
            img_dir,
            filename,
        )

        try:
            pred, confidences, avg_conf = (
                model.predict(image_path)
            )

        except Exception as exc:
            print(
                f"[warning] 跳过 {filename}: "
                f"{exc}"
            )
            continue

        recovered, candidates, low_positions = (
            position_constrained_recover(
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

        low_position_text = ",".join(
            str(i)
            for i in sorted(low_positions)
        )

        candidate_text = " | ".join(
            f"{word}:{cost:.4f}"
            for word, cost, _, _ in candidates
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
            "low_positions": low_position_text,
            "candidates": candidate_text,
        })

        if (
            not recovered_ok
            and len(wrong_examples) < 30
        ):
            wrong_examples.append({
                "filename": filename,
                "gt": gt,
                "pred": pred,
                "recovered": recovered,
                "avg_conf": avg_conf,
                "confidences": confidences,
                "low_positions": sorted(
                    low_positions
                ),
                "candidates": candidates,
            })

        if total % 200 == 0:
            print(
                f"[level {level}] [{total}] "
                f"raw={raw_correct / total:.4f} "
                f"position={recovered_correct / total:.4f} "
                f"changed={changed} "
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

def print_wrong_examples(
    wrong_examples: Sequence[Dict],
) -> None:
    print("\nWrong examples:\n")

    for case in wrong_examples:
        print(
            f"{case['filename']} | "
            f"GT={case['gt']} | "
            f"PARSeq={case['pred']} | "
            f"Position={case['recovered']} | "
            f"avg_conf={case['avg_conf']:.4f}"
        )

        print(
            "  low positions:",
            case["low_positions"],
        )

        if case["confidences"]:
            confidence_text = " ".join(
                f"{i}:{confidence:.3f}"
                for i, confidence in enumerate(
                    case["confidences"]
                )
            )

            print(
                "  confidences:",
                confidence_text,
            )

        if case["candidates"]:
            print("  candidates:")

            for (
                word,
                cost,
                replace_count,
                overlap,
            ) in case["candidates"]:
                print(
                    f"    {word:20s} "
                    f"cost={cost:.4f} "
                    f"replace={replace_count} "
                    f"overlap={overlap:.2f}"
                )

def save_csv(rows: Sequence[Dict]) -> None:
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
        "low_positions",
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
        "same-length recovery:",
        True,
    )

    gt_data = load_gt(GT_TXT)

    print("GT samples:", len(gt_data))

    lexicon = build_lexicon(gt_data)

    if not lexicon:
        raise RuntimeError(
            "词表为空。请启用USE_GT_LEXICON，"
            "或设置EXTERNAL_LEXICON_TXT。"
        )

    print("lexicon size:", len(lexicon))

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

    model = PARSeqPositionConstrainedOCR(
        DEVICE
    )

    all_csv_rows = []

    print("=" * 72)
    print(
        "PARSeq + Position-Constrained "
        "Character Recovery V2"
    )
    print("=" * 72)

    for level in LEVELS:
        print("\n" + "=" * 72)
        print("LEVEL:", level)

        result = evaluate_level(
            model,
            gt_data,
            length_index,
            level,
        )

        print("-" * 72)
        print("total:", result["total"])
        print(
            "PARSeq correct:",
            result["raw_correct"],
        )
        print(
            "Position correct:",
            result["recovered_correct"],
        )
        print(
            "PARSeq accuracy:",
            result["raw_acc"],
        )
        print(
            "Position accuracy:",
            result["recovered_acc"],
        )
        print("gain:", result["gain"])
        print("changed:", result["changed"])
        print("improved:", result["improved"])
        print("damaged:", result["damaged"])
        print(
            "unchanged wrong:",
            result["unchanged_wrong"],
        )
        print("-" * 72)

        print_wrong_examples(
            result["wrong_examples"]
        )

        all_csv_rows.extend(
            result["csv_rows"]
        )

    save_csv(all_csv_rows)

    print("\n" + "=" * 72)
    print("DONE")
    print("saved:", OUTPUT_CSV)
    print("=" * 72)

if __name__ == "__main__":
    main()
