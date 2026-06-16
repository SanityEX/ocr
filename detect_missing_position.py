import difflib

GT = "FAIRPRICE"
PRED = "HAPRICE"

def detect_missing(gt, pred):
    matcher = difflib.SequenceMatcher(None, gt, pred)
    results = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            results.append({
                "type": "missing",
                "gt_pos": i1,
                "missing_text": gt[i1:i2],
                "pred_pos": j1
            })

        elif tag == "replace":
            results.append({
                "type": "replace",
                "gt_pos": i1,
                "gt_text": gt[i1:i2],
                "pred_pos": j1,
                "pred_text": pred[j1:j2]
            })

        elif tag == "insert":
            results.append({
                "type": "extra",
                "pred_pos": j1,
                "extra_text": pred[j1:j2]
            })

    return results

def main():
    print("GT:", GT)
    print("Pred:", PRED)

    results = detect_missing(GT, PRED)

    print("=" * 50)

    for r in results:
        print(r)

if __name__ == "__main__":
    main()
