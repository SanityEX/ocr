# Project Structure / プロジェクト構成

## 日本語

### 主要な実行ファイル

```text
run_real_image_test.py          バッチ認識と可視化
test_one_image.py               単一画像認識
webcam_realtime_ocr.py          Web カメラ認識
manage_user_lexicon.py          ユーザー語彙管理
compare_adaptive_lexicon.py     適応語彙の効果比較
analyze_failure_cases.py        失敗ケース分析
```

### 認識処理の中心

```text
predict_target90_recovery.py    OCR、補完、候補ランキング、拒否判定
model_attention_ocr.py          Attention OCR モデル
evaluate_attention_occlusion.py 評価と辞書ユーティリティ
recovery_gate_common.py         編集距離と補完補助
utils.py                        文字集合などの共通定義
```

### モデルと語彙

```text
best_attention_heavy_realstyle_acc.pth
data_target90_fixed_1k/domain_lexicon.txt
english_lexicon.txt
common_english_words_extra.txt
common_english_wordfreq_50k.txt
user_adaptive_lexicon.txt
```

実行時辞書の優先順位:

```text
1. labels.txt / custom_words.txt / user_adaptive_lexicon.txt
2. project lexicon
3. common English lexicon
```

### データと出力

```text
real_demo_images/                 主要デモ画像
real_demo_results/                40 枚デモの結果
real_demo_large_test/             120 枚生成テストセット
real_demo_large_results/          120 枚テスト結果
online_real_iiit5k_sample_200/    実画像サンプル
online_real_iiit5k_results_200/   実画像テスト結果
adaptive_demo_test/               適応語彙検証用画像
adaptive_demo_comparison/         適応語彙比較結果
failure_analysis/                 失敗ケース分析結果
webcam_runtime_results/           カメラ認識ログ
```

### 発表で見せるとよいファイル

```text
README.md
docs/DEMO_GUIDE.md
docs/PROJECT_STATUS.md
real_demo_results/real_predictions_visual.png
real_demo_results/real_predictions_dashboard.html
adaptive_demo_comparison/adaptive_comparison.html
failure_analysis/failure_dashboard.html
```

---

## English

### Main Entry Points

```text
run_real_image_test.py          Batch recognition and visualization
test_one_image.py               Single-image recognition
webcam_realtime_ocr.py          Webcam recognition
manage_user_lexicon.py          User lexicon management
compare_adaptive_lexicon.py     Adaptive lexicon comparison
analyze_failure_cases.py        Failure-case analysis
```

### Core Recognition Logic

```text
predict_target90_recovery.py    OCR, completion, candidate ranking, and rejection logic
model_attention_ocr.py          Attention OCR model
evaluate_attention_occlusion.py Evaluation and lexicon utilities
recovery_gate_common.py         Edit-distance and recovery helpers
utils.py                        Shared vocabulary definitions
```

### Models and Lexicons

```text
best_attention_heavy_realstyle_acc.pth
data_target90_fixed_1k/domain_lexicon.txt
english_lexicon.txt
common_english_words_extra.txt
common_english_wordfreq_50k.txt
user_adaptive_lexicon.txt
```

Runtime lexicon priority:

```text
1. labels.txt / custom_words.txt / user_adaptive_lexicon.txt
2. project lexicon
3. common English lexicon
```

### Data and Outputs

```text
real_demo_images/                 Main demo images
real_demo_results/                40-image demo results
real_demo_large_test/             120-image generated test set
real_demo_large_results/          120-image test results
online_real_iiit5k_sample_200/    Real-scene sample images
online_real_iiit5k_results_200/   Real-scene test results
adaptive_demo_test/               Adaptive lexicon validation images
adaptive_demo_comparison/         Adaptive lexicon comparison results
failure_analysis/                 Failure-case analysis results
webcam_runtime_results/           Webcam recognition logs
```

### Recommended Files To Present

```text
README.md
docs/DEMO_GUIDE.md
docs/PROJECT_STATUS.md
real_demo_results/real_predictions_visual.png
real_demo_results/real_predictions_dashboard.html
adaptive_demo_comparison/adaptive_comparison.html
failure_analysis/failure_dashboard.html
```
