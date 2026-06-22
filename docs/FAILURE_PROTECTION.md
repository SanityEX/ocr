# Failure Protection / 失敗保護

## 日本語

### 目的

辞書補完は OCR の誤りを修正できますが、遮蔽が強い場合には誤った単語へ過補完する危険があります。そのため、このプロジェクトでは信頼できる場合だけ結果を自動採用し、危険な場合は `uncertain` として確認対象にします。

### 主な保護理由

```text
accepted
```

信頼度チェックを通過した結果です。

```text
ambiguous_candidates
```

複数の候補が近すぎるため、最上位候補を完全には信頼しません。

```text
large_correction_distance
```

OCR の生出力と補完後の単語が大きく異なります。

```text
large_gap_prefix_rescue_manual_check
```

優先語彙による救援候補はありますが、OCR 出力との差が大きいため確認対象です。

```text
low_ocr_confidence
```

OCR 全体の信頼度が低い状態です。

```text
low_character_confidence
```

一部の文字位置が不確かです。例: `QU?LITY`

```text
no_reliable_candidate
```

信頼できる辞書候補が見つかりません。

### 現在の分析結果

396 件の結果を分析したところ、問題ケースは 34 件でした。

```text
problem_rate: 0.0859
```

主な理由:

```text
ambiguous_candidates:                 17
large_correction_distance:            14
large_gap_prefix_rescue_manual_check:  2
adaptive_not_correct:                  1
```

### 方針

このシステムは、誤った単語を自信ありとして出すよりも、確認が必要な結果として表示することを優先します。これにより、採用された結果の精度を高く保ちます。

---

## English

### Purpose

Lexicon completion can correct OCR errors, but it can also over-correct when the image is heavily occluded. This project therefore accepts a result only when it is reliable enough. Risky results are marked as `uncertain` for manual confirmation.

### Main Protection Reasons

```text
accepted
```

The result passed the confidence checks.

```text
ambiguous_candidates
```

Several candidates are too close, so the top candidate is not trusted enough.

```text
large_correction_distance
```

The completed word is too different from the raw OCR output.

```text
large_gap_prefix_rescue_manual_check
```

A priority-word rescue candidate exists, but the gap from raw OCR is still large.

```text
low_ocr_confidence
```

The average OCR confidence is low.

```text
low_character_confidence
```

Some character positions are uncertain, such as `QU?LITY`.

```text
no_reliable_candidate
```

No reliable lexicon candidate was found.

### Current Analysis

Across 396 analyzed rows, 34 rows were problem cases.

```text
problem_rate: 0.0859
```

Main reasons:

```text
ambiguous_candidates:                 17
large_correction_distance:            14
large_gap_prefix_rescue_manual_check:  2
adaptive_not_correct:                  1
```

### Policy

The system prioritizes reliability over forcing an answer. It is better to show a result as confirmation-required than to accept a wrong completion with high confidence.
