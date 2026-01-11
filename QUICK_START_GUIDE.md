# Quick Start Guide - Dissertation Ready

## ⚡ Quick Commands

### 1. Train Models (With All Dissertation Metrics)
```bash
# Train BiLSTM (recommended settings)
python train.py --model bilstm --epochs 20 --batch_size 8 --augment

# Train GRU
python train.py --model gru --epochs 20 --batch_size 8 --augment

# Train without augmentation
python train.py --model bilstm --epochs 20 --no_augment
```

### 2. Evaluate Models (All Metrics)
```bash
# Compare both models
python evaluate_model.py --compare

# Evaluate single model
python evaluate_model.py --model bilstm --split test

# Evaluate on validation set
python evaluate_model.py --model bilstm --split val
```

### 3. Analyze & Compare (Complete Analysis)
```bash
# Full BiLSTM vs GRU comparison with plots
python analyze_models.py
```

### 4. Real-time Demo
```bash
# Webcam demo
python realtime_lipreading.py --model bilstm

# Video file demo
python demo_video_file.py --model bilstm --source path/to/video.mp4
```

---

## 📊 What Metrics You Get

### Training Output (Every Epoch):
- Top-1 Accuracy (%)
- Top-3 Accuracy (%)
- Top-5 Accuracy (%)
- Word Error Rate (WER %)
- Character Error Rate (CER %)
- Per-class Accuracy
- Training/Validation Loss

### Final Test Evaluation:
- All above metrics PLUS:
- WER details (total words, errors, correct)
- CER details (total characters, character errors)
- Per-class breakdown

### Model Comparison:
- Side-by-side BiLSTM vs GRU
- Model complexity (parameters, size)
- Inference speed (latency, FPS)
- Accuracy comparison
- Error rate comparison
- 4-panel visualization plot

---

## 📁 Output Files Generated

### After Training:
```
saved_models/
├── cnn_bilstm.pth                    # Trained BiLSTM weights
├── cnn_gru.pth                       # Trained GRU weights
├── bilstm_training_history.json      # Complete training metrics
├── gru_training_history.json         # Complete training metrics
└── labels.json                       # Class mappings
```

### After Evaluation:
```
saved_models/
├── bilstm_test_evaluation.json           # Comprehensive metrics
├── gru_test_evaluation.json              # Comprehensive metrics
├── bilstm_test_confusion_matrix.png      # Confusion matrix plot
└── gru_test_confusion_matrix.png         # Confusion matrix plot
```

### After Analysis:
```
saved_models/
├── comparison_report.json            # Full comparison (JSON)
└── model_comparison.png              # 4-panel visualization
```

---

## 🎯 For Your Dissertation

### Chapter 4 (Results) - Run These:
```bash
# 1. Train both models
python train.py --model bilstm --epochs 20
python train.py --model gru --epochs 20

# 2. Comprehensive evaluation
python evaluate_model.py --compare

# 3. Generate comparison plots
python analyze_models.py
```

### Chapter 5 (Discussion) - Use These Files:
1. **Table 5.1 (Accuracy Metrics)**
   - Source: `saved_models/comparison_report.json`
   - Metrics: Top-1, Top-3, Top-5, WER, CER

2. **Table 5.2 (Model Complexity)**
   - Source: `saved_models/comparison_report.json`
   - Metrics: Parameters, Model Size

3. **Table 5.3 (Inference Performance)**
   - Source: `saved_models/comparison_report.json`
   - Metrics: Latency (mean, std), FPS

4. **Figure 5.1 (Comparison Plot)**
   - Source: `saved_models/model_comparison.png`
   - Contains: 4-panel comparison visualization

5. **Figure 5.2 (Confusion Matrices)**
   - Source: `saved_models/*_confusion_matrix.png`
   - One for BiLSTM, one for GRU

6. **Figure 5.3 (Training Curves)**
   - Source: `saved_models/*_training_history.json`
   - Plot with matplotlib/pandas

---

## 💡 Pro Tips

### For Best Results:
```bash
# Use these exact settings for dissertation
python train.py --model bilstm --epochs 20 --batch_size 8 --lr 0.001 --dropout 0.5 --weight_decay 0.0001 --augment
python train.py --model gru --epochs 20 --batch_size 8 --lr 0.001 --dropout 0.5 --weight_decay 0.0001 --augment
```

### If GPU Memory Issues:
```bash
# Reduce batch size
python train.py --model bilstm --epochs 20 --batch_size 4
```

### For Faster Testing:
```bash
# Train with fewer epochs first
python train.py --model bilstm --epochs 5
```

---

## 🔍 Verification Checklist

Before using for dissertation:

1. **Training Complete:**
   - [ ] BiLSTM model trained (20 epochs)
   - [ ] GRU model trained (20 epochs)
   - [ ] Both `.pth` files in `saved_models/`
   - [ ] Training history JSON files created

2. **Evaluation Done:**
   - [ ] Run `python evaluate_model.py --compare`
   - [ ] Evaluation JSON files created
   - [ ] Confusion matrices generated

3. **Analysis Complete:**
   - [ ] Run `python analyze_models.py`
   - [ ] `comparison_report.json` created
   - [ ] `model_comparison.png` created

4. **Metrics Present:**
   - [ ] Top-1, Top-3, Top-5 Accuracy ✓
   - [ ] WER and CER ✓
   - [ ] Per-class metrics ✓
   - [ ] Confusion matrices ✓
   - [ ] Inference speed ✓

---

## 📈 Sample Output Interpretation

### Training Output:
```
Epoch 20/20
Train - Acc: 95.80% | Loss: 0.1234 | Time: 125.3s
Val   - Acc: 88.70% | Loss: 0.3456 | Time: 12.5s

--- Validation Metrics (Dissertation Requirements) ---
Top-1 Accuracy: 88.70%   ← Main accuracy metric
Top-3 Accuracy: 96.40%   ← 96.4% of time correct word in top-3
Top-5 Accuracy: 100.00%  ← All predictions have correct word in top-5
Word Error Rate (WER): 11.30%    ← 11.3% of words wrong
Character Error Rate (CER): 7.52%  ← 7.52% character-level errors
```

### Comparison Output:
```
🏆 Best Model (Top-1 Accuracy): BiLSTM (+2.40%)
⚡ Fastest Model (Latency): BiLSTM (2.07 ms faster)
```
This means:
- BiLSTM is 2.40% more accurate
- BiLSTM is also 2.07ms faster per inference
- **Conclusion:** BiLSTM is better for this task

---

## 🆘 Troubleshooting

### "Model file not found"
```bash
# Train the model first
python train.py --model bilstm --epochs 20
```

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --model bilstm --batch_size 4
```

### "No module named 'src.utils'"
```bash
# Make sure you're in the correct directory
cd /home/unik/Desktop/huhu
python train.py ...
```

### Want to see augmentation in action?
The augmentation is automatically applied during training when `--augment` flag is used (default: enabled).

---

## 📚 All Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `train.py` | Train models with all metrics | `python train.py --model bilstm` |
| `evaluate_model.py` | Comprehensive evaluation | `python evaluate_model.py --compare` |
| `analyze_models.py` | Full model comparison | `python analyze_models.py` |
| `realtime_lipreading.py` | Webcam demo | `python realtime_lipreading.py` |
| `demo_video_file.py` | Video file demo | `python demo_video_file.py --model bilstm` |
| `diagnose_problem.py` | System diagnostics | `python diagnose_problem.py` |

---

## ✅ Ready for Dissertation Submission!

All requirements from your dissertation report are now implemented:
- ✅ Data Augmentation (5 techniques)
- ✅ WER and CER metrics
- ✅ Top-1, Top-3, Top-5 Accuracy
- ✅ Per-class Performance
- ✅ Confusion Matrices
- ✅ Model Complexity Analysis
- ✅ Inference Speed Measurement
- ✅ Comprehensive Comparison

---

**Need help?** Check `IMPLEMENTATION_SUMMARY.md` for detailed documentation.
