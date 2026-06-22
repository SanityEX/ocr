# Demo Guide / デモ手順

## 日本語

### 1. バッチ認識

```powershell
python run_real_image_test.py
```

確認する出力:

```text
real_demo_results/real_predictions_visual.png
real_demo_results/real_predictions_dashboard.html
```

### 2. 単一画像認識

```powershell
python test_one_image.py --image real_demo_images\33.png
```

結果には OCR の生出力、補完後の単語、候補語、確認保護理由が表示されます。

### 3. Web カメラ認識

```powershell
python webcam_realtime_ocr.py --fast-mode --auto-roi --save-hard-cases
```

主な操作:

```text
Q      終了
S      現在の ROI を保存
U      現在の結果をユーザー語彙に追加
Space  その場で認識
T      ライブ/写真モード切替
A      自動 ROI 切替
```

### 4. ユーザー適応語彙のデモ

```powershell
python generate_adaptive_demo_set.py
python compare_adaptive_lexicon.py --image-dir adaptive_demo_test --labels adaptive_demo_test\labels.txt --user-words adaptive_demo_test\adaptive_words.txt --output-dir adaptive_demo_comparison --disable-label-words
python make_adaptive_demo_preview.py
```

確認する出力:

```text
adaptive_demo_test_preview.png
adaptive_demo_comparison/adaptive_comparison.html
adaptive_demo_comparison/adaptive_comparison_summary.txt
```

### 5. 失敗ケース分析

```powershell
python analyze_failure_cases.py
```

確認する出力:

```text
failure_analysis/failure_dashboard.html
failure_analysis/failure_summary.txt
failure_analysis/problem_cases.csv
```

### 説明する処理の流れ

```text
入力画像
-> OCR モデルが文字列を予測
-> 文字信頼度を分析
-> 辞書から候補語を生成
-> 候補ランキングで補完語を選択
-> 拒否ゲートで accepted / uncertain を判定
-> 結果と候補語を可視化
```

重要な点は、システムが常に答えを強制しないことです。危険な補完は `uncertain` として確認対象にします。

---

## English

### 1. Batch Recognition

```powershell
python run_real_image_test.py
```

Outputs to show:

```text
real_demo_results/real_predictions_visual.png
real_demo_results/real_predictions_dashboard.html
```

### 2. Single-Image Recognition

```powershell
python test_one_image.py --image real_demo_images\33.png
```

The result shows raw OCR output, completed word, candidates, and the protection reason.

### 3. Webcam Recognition

```powershell
python webcam_realtime_ocr.py --fast-mode --auto-roi --save-hard-cases
```

Main controls:

```text
Q      Quit
S      Save current ROI
U      Add current result to user lexicon
Space  Run recognition once
T      Toggle live/photo mode
A      Toggle auto ROI
```

### 4. User-Adaptive Lexicon Demo

```powershell
python generate_adaptive_demo_set.py
python compare_adaptive_lexicon.py --image-dir adaptive_demo_test --labels adaptive_demo_test\labels.txt --user-words adaptive_demo_test\adaptive_words.txt --output-dir adaptive_demo_comparison --disable-label-words
python make_adaptive_demo_preview.py
```

Outputs to show:

```text
adaptive_demo_test_preview.png
adaptive_demo_comparison/adaptive_comparison.html
adaptive_demo_comparison/adaptive_comparison_summary.txt
```

### 5. Failure-Case Analysis

```powershell
python analyze_failure_cases.py
```

Outputs to show:

```text
failure_analysis/failure_dashboard.html
failure_analysis/failure_summary.txt
failure_analysis/problem_cases.csv
```

### Pipeline To Explain

```text
Input image
-> OCR model predicts raw text
-> Character confidence is analyzed
-> Lexicon candidates are generated
-> Candidate ranking selects the completed word
-> Rejection gate decides accepted / uncertain
-> Result and candidates are visualized
```

The key point is that the system does not always force an answer. Risky completions are marked as `uncertain` for manual confirmation.
