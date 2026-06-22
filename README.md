# Occluded Word OCR and Completion

## 日本語

本プロジェクトは、部分的に隠れた英単語画像を対象に、OCR による文字認識と辞書ベースの単語補完を組み合わせた卒業研究用プロトタイプです。通常の OCR 出力だけでなく、候補語、信頼度、確認が必要なケース、カメラ認識、ユーザー適応語彙も扱います。

Zhang Yiheng

### 主な機能

- 英単語画像向け Attention OCR モデル
- 部分遮蔽された単語の補完
- プロジェクト語彙、一般英単語、ユーザー語彙を統合した実行時辞書
- 候補語ランキングと候補ソース表示
- 文字単位の信頼度表示
- 誤補完を避けるための拒否・確認保護
- 画像前処理リトライ
- 単一画像テスト、バッチテスト、Web カメラ認識
- ユーザー適応語彙の追加・削除・検索
- 失敗ケース分析と HTML ダッシュボード

### 現在の結果

40 枚のデモ画像:

```text
final_acc:    1.0000
accepted_acc: 1.0000
reject_rate:  0.1500
```

120 枚の生成遮蔽テスト画像:

```text
final_acc:    0.9667
accepted_acc: 1.0000
reject_rate:  0.0333
```

200 枚の実画像サンプル:

```text
final_acc:    0.9050
accepted_acc: 1.0000
reject_rate:  0.1150
```

ユーザー適応語彙の検証用セット:

```text
base_final_acc:     0.0000
adaptive_final_acc: 0.9722
accepted_acc:       1.0000
improved:           35 / 36
regressed:          0
```

このシステムは常に強制的に答えを出すわけではありません。OCR 出力と補完候補の差が大きい場合や候補が曖昧な場合は、`uncertain` として確認対象にします。

### 使い方

仮想環境を有効化:

```powershell
.\.venv\Scripts\Activate.ps1
```

バッチテスト:

```powershell
python run_real_image_test.py
```

単一画像テスト:

```powershell
python test_one_image.py --image real_demo_images\33.png
```

Web カメラ認識:

```powershell
python webcam_realtime_ocr.py --fast-mode --auto-roi --save-hard-cases
```

PowerShell ヘルパー:

```powershell
.\scripts\run_demo.ps1 -Mode batch
.\scripts\run_demo.ps1 -Mode single -Image real_demo_images\33.png
.\scripts\run_demo.ps1 -Mode webcam
```

### ユーザー適応語彙

個人語彙ファイル:

```text
user_adaptive_lexicon.txt
```

語彙管理:

```powershell
python manage_user_lexicon.py stats
python manage_user_lexicon.py add Wednesday OCRPROJECTX
python manage_user_lexicon.py search Wednesday
python manage_user_lexicon.py remove WRONGWORD
python manage_user_lexicon.py clean
```

適応語彙の効果比較:

```powershell
python generate_adaptive_demo_set.py
python compare_adaptive_lexicon.py --image-dir adaptive_demo_test --labels adaptive_demo_test\labels.txt --user-words adaptive_demo_test\adaptive_words.txt --output-dir adaptive_demo_comparison --disable-label-words
python make_adaptive_demo_preview.py
```

失敗ケース分析:

```powershell
python analyze_failure_cases.py
```

### 主要ファイル

```text
run_real_image_test.py          バッチテストと可視化
test_one_image.py               単一画像テスト
webcam_realtime_ocr.py          Web カメラ認識
predict_target90_recovery.py    OCR、補完、信頼度、拒否判定
manage_user_lexicon.py          ユーザー語彙管理
compare_adaptive_lexicon.py     適応語彙あり/なしの比較
analyze_failure_cases.py        失敗ケース分析
model_attention_ocr.py          Attention OCR モデル
train_qc_occlusion_finetune_v2.py 学習・デコード補助
```

### 出力例

```text
real_demo_results/real_predictions.csv
real_demo_results/real_predictions_visual.png
real_demo_results/real_predictions_dashboard.html
webcam_runtime_results/webcam_predictions.csv
failure_analysis/failure_dashboard.html
adaptive_demo_comparison/adaptive_comparison.html
```

### 注意

リポジトリには研究過程の実験スクリプトやチェックポイントも含まれています。通常のデモでは、上記の主要ファイルを中心に使用します。

---

## English

This project is a graduation-project prototype for recognizing partially occluded English word images. It combines an Attention-based OCR model with lexicon-based word completion, candidate ranking, confidence display, rejection protection, webcam recognition, and a user-adaptive lexicon.

Student: 3CDA1101 Zhang Yiheng

### Main Features

- Attention OCR model for English word images
- Word completion for partially occluded text
- Runtime lexicon from project words, common English words, and user-adaptive words
- Candidate ranking with source information
- Character-level confidence diagnostics
- Conservative rejection gate to avoid overconfident wrong completions
- Image preprocessing retry for hard samples
- Batch testing, single-image testing, and webcam recognition
- User-adaptive lexicon add/search/remove/clean workflow
- Failure-case analysis and HTML dashboards

### Current Results

40-image demo set:

```text
final_acc:    1.0000
accepted_acc: 1.0000
reject_rate:  0.1500
```

120-image generated occlusion test set:

```text
final_acc:    0.9667
accepted_acc: 1.0000
reject_rate:  0.0333
```

200-image real-scene sample set:

```text
final_acc:    0.9050
accepted_acc: 1.0000
reject_rate:  0.1150
```

Adaptive lexicon validation set:

```text
base_final_acc:     0.0000
adaptive_final_acc: 0.9722
accepted_acc:       1.0000
improved:           35 / 36
regressed:          0
```

The system does not always force an answer. If the OCR output and the completion candidate are too different, or if the candidates are ambiguous, the result is marked as `uncertain` for manual confirmation.

### Quick Start

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the batch demo:

```powershell
python run_real_image_test.py
```

Run one image:

```powershell
python test_one_image.py --image real_demo_images\33.png
```

Run webcam recognition:

```powershell
python webcam_realtime_ocr.py --fast-mode --auto-roi --save-hard-cases
```

PowerShell helper:

```powershell
.\scripts\run_demo.ps1 -Mode batch
.\scripts\run_demo.ps1 -Mode single -Image real_demo_images\33.png
.\scripts\run_demo.ps1 -Mode webcam
```

### User-Adaptive Lexicon

User lexicon file:

```text
user_adaptive_lexicon.txt
```

Manage the lexicon:

```powershell
python manage_user_lexicon.py stats
python manage_user_lexicon.py add Wednesday OCRPROJECTX
python manage_user_lexicon.py search Wednesday
python manage_user_lexicon.py remove WRONGWORD
python manage_user_lexicon.py clean
```

Compare results with and without the adaptive lexicon:

```powershell
python generate_adaptive_demo_set.py
python compare_adaptive_lexicon.py --image-dir adaptive_demo_test --labels adaptive_demo_test\labels.txt --user-words adaptive_demo_test\adaptive_words.txt --output-dir adaptive_demo_comparison --disable-label-words
python make_adaptive_demo_preview.py
```

Analyze failure cases:

```powershell
python analyze_failure_cases.py
```

### Important Files

```text
run_real_image_test.py          Batch test and visualization
test_one_image.py               Single-image test entry
webcam_realtime_ocr.py          Webcam recognition demo
predict_target90_recovery.py    OCR prediction, completion, confidence, and rejection logic
manage_user_lexicon.py          User lexicon management
compare_adaptive_lexicon.py     Comparison with and without adaptive lexicon
analyze_failure_cases.py        Failure-case analysis
model_attention_ocr.py          Attention OCR model
train_qc_occlusion_finetune_v2.py Training / decoding helper
```

### Output Examples

```text
real_demo_results/real_predictions.csv
real_demo_results/real_predictions_visual.png
real_demo_results/real_predictions_dashboard.html
webcam_runtime_results/webcam_predictions.csv
failure_analysis/failure_dashboard.html
adaptive_demo_comparison/adaptive_comparison.html
```

### Notes

The repository also contains historical experiment scripts and model checkpoints. For normal demonstration, focus on the important files listed above.
