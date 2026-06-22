# Project Status / プロジェクト状況

## 日本語

### 完了した機能

- 英単語画像向け OCR
- 部分遮蔽単語の補完
- 一般英単語、プロジェクト語彙、ユーザー語彙を統合した実行時辞書
- 候補語ランキング
- 文字単位の信頼度診断
- 誤補完を避ける拒否・確認保護
- 画像前処理リトライ
- 単一画像テスト
- バッチテスト
- Web カメラ認識
- ユーザー適応語彙の管理
- 適応語彙あり/なしの比較実験
- 失敗ケース分析
- PNG / HTML 可視化

### 評価結果

40 枚のデモ画像:

```text
total:        40
accepted:     34
reject_rate:  0.1500
raw_acc:      0.3000
final_acc:    1.0000
accepted_acc: 1.0000
```

120 枚の生成遮蔽テスト画像:

```text
total:        120
accepted:     116
reject_rate:  0.0333
raw_acc:      0.6750
final_acc:    0.9667
accepted_acc: 1.0000
```

200 枚の実画像サンプル:

```text
total:        200
accepted:     177
reject_rate:  0.1150
raw_acc:      0.6950
final_acc:    0.9050
accepted_acc: 1.0000
```

ユーザー適応語彙の検証:

```text
total:              36
base_final_acc:     0.0000
adaptive_final_acc: 0.9722
adaptive_accepted_acc: 1.0000
improved:           35
regressed:          0
```

失敗ケース分析:

```text
total:         396
problem_count: 34
problem_rate:  0.0859
```

主な失敗理由:

```text
ambiguous_candidates:                 17
large_correction_distance:            14
large_gap_prefix_rescue_manual_check:  2
adaptive_not_correct:                  1
```

### 残っている課題

- 実カメラ画像をさらに増やす
- ROI 検出をより安定させる
- 候補語スコアをより信頼度に近づける
- 短い単語の誤補完保護をさらに強化する
- 日語文字や多語文脈への拡張を検討する

---

## English

### Completed Features

- OCR for English word images
- Completion for partially occluded words
- Runtime lexicon combining common English words, project words, and user-adaptive words
- Candidate ranking
- Character-level confidence diagnostics
- Rejection and confirmation protection against wrong overconfident completions
- Image preprocessing retry
- Single-image testing
- Batch testing
- Webcam recognition
- User-adaptive lexicon management
- Comparison with and without adaptive lexicon
- Failure-case analysis
- PNG / HTML visualization

### Evaluation Results

40-image demo set:

```text
total:        40
accepted:     34
reject_rate:  0.1500
raw_acc:      0.3000
final_acc:    1.0000
accepted_acc: 1.0000
```

120-image generated occlusion test set:

```text
total:        120
accepted:     116
reject_rate:  0.0333
raw_acc:      0.6750
final_acc:    0.9667
accepted_acc: 1.0000
```

200-image real-scene sample set:

```text
total:        200
accepted:     177
reject_rate:  0.1150
raw_acc:      0.6950
final_acc:    0.9050
accepted_acc: 1.0000
```

Adaptive lexicon validation:

```text
total:              36
base_final_acc:     0.0000
adaptive_final_acc: 0.9722
adaptive_accepted_acc: 1.0000
improved:           35
regressed:          0
```

Failure-case analysis:

```text
total:         396
problem_count: 34
problem_rate:  0.0859
```

Main failure reasons:

```text
ambiguous_candidates:                 17
large_correction_distance:            14
large_gap_prefix_rescue_manual_check:  2
adaptive_not_correct:                  1
```

### Remaining Work

- Add more real camera-captured images
- Improve ROI detection stability
- Make candidate scores closer to calibrated confidence
- Further strengthen short-word false-completion protection
- Explore Japanese text and multi-word context extension
