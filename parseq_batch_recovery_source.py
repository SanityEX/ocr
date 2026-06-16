import argparse
import csv
from pathlib import Path

from eval_parseq_safe_hybrid_recovery_v4 import (
    DEVICE,
    PARSeqSafeHybridOCR,
    build_length_index,
    safe_hybrid_recover,
)
from batch_predict_word_recovery import collect_images, read_labels
from predict_word_recovery import DEFAULT_LEXICON, load_merged_lexicon
from recovery_gate_common import normalize_text

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "data_recovery" / "parseq_recovery_source.csv"

def infer_level(filename: str, fallback: str) -> str:
    name = filename.lower()

    for level in ("clean", "light", "medium", "heavy", "full_char"):
        if level in name:
            return level

    return fallback

def main():
    parser = argparse.ArgumentParser(
        description="Run PARSeq + Safe Hybrid V4 on a folder and save source rows for recovery-ranker data building."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--custom-lexicon", type=Path, action="append", default=[])
    parser.add_argument("--default-level", default="custom")
    args = parser.parse_args()

    images = collect_images(args.image_dir)
    labels = read_labels(args.labels)

    if not images:
        raise RuntimeError(f"no images found in {args.image_dir}")

    lexicon = load_merged_lexicon(args.lexicon, args.custom_lexicon)
    lexicon_set = set(lexicon)
    length_index = build_length_index(lexicon)
    ocr = PARSeqSafeHybridOCR(DEVICE)

    rows = []

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}", flush=True)

        gt = normalize_text(labels.get(image_path.name, ""))
        if not gt:
            continue

        raw, confidences, avg_conf = ocr.predict(str(image_path))
        recovered, candidates, reason = safe_hybrid_recover(
            raw,
            confidences,
            lexicon_set,
            length_index,
        )

        rows.append({
            "level": infer_level(image_path.name, args.default_level),
            "filename": image_path.name,
            "gt": gt,
            "raw_prediction": raw,
            "recovered_prediction": recovered,
            "raw_correct": int(raw == gt),
            "recovered_correct": int(recovered == gt),
            "average_confidence": f"{avg_conf:.6f}",
            "confidence_list": ",".join(f"{value:.4f}" for value in confidences),
            "recovery_reason": reason,
            "candidates": " | ".join(
                f"{candidate.word}:{candidate.edit_type}:{candidate.score:.4f}:{candidate.description}"
                for candidate in candidates
            ),
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

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

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    raw_correct = sum(row["raw_correct"] for row in rows)
    recovered_correct = sum(row["recovered_correct"] for row in rows)

    print("=" * 80)
    print("saved:", args.output_csv)
    print("rows:", total)
    print("raw_acc:", raw_correct / total if total else 0.0)
    print("v4_acc:", recovered_correct / total if total else 0.0)

if __name__ == "__main__":
    main()
