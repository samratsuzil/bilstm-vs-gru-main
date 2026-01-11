# 🎓 Lip Reading System: BiLSTM vs GRU Comparison

A comprehensive real-time visual speech recognition (VSR) system comparing Bi-LSTM and GRU architectures for lip reading, with complete training pipeline, evaluation tools, and assistive communication capabilities.

---

## 📖 Table of Contents

1. [Quick Start](#quick-start)
2. [Problem & Solution](#problem--solution)
3. [System Overview](#system-overview)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Proposal Alignment](#project-proposal-alignment)
7. [Technical Details](#technical-details)
8. [Troubleshooting](#troubleshooting)
9. [File Structure](#file-structure)
10. [References](#references)

---

## 🚀 Quick Start

### **3-Step Setup**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (REQUIRED - only 1 epoch before, now 20 epochs)
python train.py --model bilstm --epochs 20
python train.py --model gru --epochs 20

# 3. Test
python demo_video_file.py --model bilstm
```

### **Verify Setup**

```bash
# Check your environment
bash setup_check.sh

# Diagnose any issues
python diagnose_problem.py
```

---

## ⚠️ Problem & Solution

#### Root Causes:

| Issue | Cause | Impact |
|-------|-------|--------|
| **Insufficient Training** | Less epoch | Model didn't learn to differentiate classes |
| **Sequence Length Mismatch** | Frame mismatch to training | Input shape incompatibility |
| **No Lip Detection** | Full frame instead of lip region | Too much background noise |
| **Preprocessing Mismatch** | Different preprocessing in train vs inference | Data distribution shift |
| **No Diagnostics** | No confidence scores | Unable to detect failures |

---

##  System Overview

### **Architecture**

```
Video Input → Face Detection → Lip Extraction → Resize (112×112)
    ↓
Frame Buffer (15 consecutive frames)
    ↓
ResNet-18 CNN → Spatial Feature Extraction (512-dim)
    ↓
Bi-LSTM or GRU → Temporal Sequence Modeling
    ↓
Fully Connected Layer → 5-Class Softmax
    ↓
Prediction + Confidence + Top-3 Results
```

### **Key Features**

- **Real-time Capability**: ~(20-30 FPS) with ~(40-50ms) latency
- **Multiple Models**: BiLSTM and GRU
- **Lip Detection**: Automatic facial landmark detection with fallback
- **Performance Metrics**: Confidence scores, FPS monitoring, inference timing
- **Dataset Support**: Multi-class system
- **Evaluation Tools**: Comprehensive model comparison and analysis
- **Assistive Technology**: Suitable for hearing-impaired communication

---

## 📋 Installation

### **1. System Requirements**

**Hardware:**
- GPU recommended (NVIDIA RTX 2060/4070+) for training
- CPU acceptable for inference
- 8GB+ RAM

**Software:**
- Python 3.8+
- CUDA 11.0+ (optional, for GPU training)

### **2. Install Dependencies**

```bash
# Clone or navigate to project directory
cd /path/to/your_project

# Install Python packages
pip install -r requirements.txt

# Optional: Install dlib for lip detection
pip install dlib

# Download face landmark model (optional but recommended)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

### **3. Verify Installation**

```bash
# Run setup checker
bash setup_check.sh

# Check Python environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📂 Dataset Structure

Organize your dataset as follows:

```
lrw/
├── PUBLIC/
│   ├── train/
│   ├── val/
│   └── test/
├── GREAT/
│   ├── train/
│   ├── val/
│   └── test/
├── PLANS/
│   ├── train/
│   ├── val/
│   └── test/
├── YOUNG/
│   ├── train/
│   ├── val/
│   └── test/
└── KILLED/
    ├── train/
    ├── val/
    └── test/
```

**Requirements:**
- Minimum 20 videos per class per split
- Recommended: 100+ videos per class
- Video format: .mp4 with clear lip movements

---

##  Usage Guide

### **Training Models**

#### **Improved Training Script (RECOMMENDED)**

```bash
# Train BiLSTM model
python train.py --model bilstm --epochs 20

# Train GRU model
python train.py --model gru --epochs 20

# Custom configuration
python train.py --model bilstm --epochs 30 --batch_size 16 --lr 0.0005
python train.py --model gru --epochs 30 --batch_size 16 --lr 0.0005
```

**Parameters:**
- `--model`: 'bilstm' or 'gru'
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 0.001)

**Output:**
```
Training history saved to saved_models/bilstm_training_history.json
Best model saved to saved_models/(cnn_bilstm.pth & cnn_gru.pth)
```

### **Real-time Inference**

#### **Advanced Demo System** (RECOMMENDED)

```bash
# Webcam demo with BiLSTM
python realtime_lipreading.py --model bilstm

# Webcam demo with GRU
python realtime_lipreading.py --model gru

# Using video file with BiLSTM
python demo_video_file.py --model bilstm --source path/to/video.mp4

# Using video file with GRU
python demo_video_file.py --model gru --source path/to/video.mp4

# Custom sequence length
# Add --seq_len 15
```

**Demo Controls:**
- **'q'**: Quit
- **'s'**: Show statistics
- **'r'**: Reset statistics

**Demo Display:**
- Real-time prediction with confidence
- Top-3 predictions
- Frame buffer status (15/15 frames)
- FPS monitor
- Device info (CPU/GPU)

#### **Simple Real-time Demo**

```bash
# Quick demonstration
python realtime_lipreading.py
```

### **Model Analysis & Comparison**

#### **Compare BiLSTM vs GRU**

```bash
# Generate comprehensive comparison
python analyze_models.py

# Outputs:
# - Accuracy comparison (overall & per-class)
# - Inference speed benchmarks
# - Model complexity analysis
# - Visualization plots
# - Detailed report
```

**Generated Files:**
- `saved_models/comparison_report.txt` - Detailed metrics
- `saved_models/model_comparison.png` - Comparison plots
- Console output - Summary statistics

#### **Analyze Trade-offs**

```bash
python analyze_tradeoff.py
```

### **Diagnosis & Troubleshooting**

```bash
# Automated problem detection
python diagnose_problem.py

# Checks:
# - Model weight integrity
# - Dataset balance
# - Training history
# - Output diversity
# - Common failure modes
```

---

##  Expected Performance For 5 Class Model

### **Accuracy Results**

| Model | Train Acc | Val Acc | Test Acc | Per-Class Avg |
|-------|-----------|---------|----------|---------------|
| BiLSTM | 95.8% | 88.7% | 87.2% | 85-90% |
| GRU | 94.2% | 86.2% | 84.8% | 82-88% |

### **Inference Performance**

| Metric | BiLSTM | GRU |
|--------|--------|-----|
| Latency (mean) | 45ms | 38ms |
| Latency (std) | ±3ms | ±2ms |
| FPS | 22 | 26 |
| Throughput | ~1.4 predictions/sec | ~1.7 predictions/sec |

### **Model Complexity**

| Metric | BiLSTM | GRU |
|--------|--------|-----|
| Parameters | 15.2M | 12.8M |
| Model Size | 58MB | 49MB |
| Memory (inference) | ~800MB | ~600MB |

### **Trade-off Summary**

| Aspect | Winner | Advantage |
|--------|--------|-----------|
| Accuracy | BiLSTM | +2.5% better |
| Speed | GRU | 15% faster |
| Efficiency | GRU | 20% smaller |
| Best For | BiLSTM | Precision-critical tasks |
| Best For | GRU | Real-time, resource-constrained |

---



## 🔧 Technical Details

### **Model Architectures**

#### **BiLSTM Model**

```python
Input: (batch, 15, 1, 112, 112)
  ↓
ResNet-18 Feature Extraction
  ↓ (batch, 15, 512)
Bidirectional LSTM (2 layers)
  ├─ Forward LSTM (hidden=256)
  └─ Backward LSTM (hidden=256)
  ↓ (batch, 15, 512)  # 256×2 from bidirectional
Mean Pooling over temporal dimension
  ↓ (batch, 512)
Fully Connected Layer (512 → 5)
  ↓
Softmax + Cross-Entropy Loss
```

#### **GRU Model**

```python
Input: (batch, 15, 1, 112, 112)
  ↓
ResNet-18 Feature Extraction
  ↓ (batch, 15, 512)
Gated Recurrent Unit (2 layers)
  └─ Unidirectional GRU (hidden=256)
  ↓ (batch, 15, 256)
Mean Pooling over temporal dimension
  ↓ (batch, 256)
Fully Connected Layer (256 → 5)
  ↓
Softmax + Cross-Entropy Loss
```

### **Training Configuration**

```python
# Hyperparameters
EPOCHS = 20              # Increased from 1 to 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
FRAME_SAMPLE = 15       # Sequence length
IMAGE_SIZE = 112        # Lip region size
OPTIMIZER = Adam        # with weight decay
SCHEDULER = ReduceLROnPlateau(patience=3, factor=0.5)
LOSS_FUNCTION = CrossEntropyLoss
DEVICE = GPU (if available) else CPU
```

### **Data Pipeline**

```python
1. Video Loading
   ├─ Read MP4 file
   ├─ Extract frames
   └─ Frame count: Variable

2. Face Detection
   ├─ Use dlib detector
   ├─ Extract facial landmarks
   └─ Fallback to center region

3. Lip Region Extraction
   ├─ Points 48-68 (mouth landmarks)
   ├─ Add margin (±30 pixels)
   └─ Fallback: Lower 30% of frame

4. Frame Sampling
   ├─ Uniformly sample 15 frames
   ├─ If < 15 frames: Repeat last frame
   └─ If > 15 frames: Linear interpolation

5. Preprocessing
   ├─ Resize to 112×112
   ├─ Convert to grayscale
   ├─ Normalize [0, 1]
   └─ Add channel dimension (1, 112, 112)

6. Stacking
   ├─ Stack 15 frames
   ├─ Shape: (15, 1, 112, 112)
   └─ Add batch dimension if needed
```

---

## 🐛 Troubleshooting

### **Model Predicts Same Class**

**Cause:** Insufficient training or dataset imbalance

**Solution:**
```bash
# Check dataset balance
python diagnose_problem.py

# Retrain with more epochs
python train.py --model bilstm --epochs 30
```

### **Low Accuracy (<70%)**

**Causes:**
- Too few training epochs
- Dataset too small
- Class imbalance
- Poor video quality

**Solutions:**
1. Increase training epochs (30-50)
2. Collect more data per class
3. Balance dataset samples
4. Check video quality and preprocessing

### **Slow Inference**

**Causes:**
- Using CPU instead of GPU
- Large batch size

**Solutions:**
```bash
# Use GPU if available

# Reduce batch size
python train.py --batch_size 4
```

### **dlib Installation Issues**

```bash
# Ubuntu/Debian
sudo apt-get install cmake
pip install dlib

# macOS
brew install cmake
pip install dlib

# If still failing, use without dlib (fallback mode works)
```

### **GPU Memory Issues**

```bash
# Reduce batch size
python train.py --batch_size 4

# Reduce epochs to test
python train.py --epochs 5

# Use CPU only
# Modify device = torch.device("cuda" -> "cpu")
```

### **Dataset Not Found**

```bash
# Verify structure
ls -la lrw/
ls -la lrw/WORD1/train/

# Should see:
# - 5 word directories: PUBLIC, GREAT, PLANS, YOUNG, KILLED
# - Each with: train/, val/, test/ subdirectories
# - Each containing .mp4 video files
```

---

## 📁 File Structure

```
HUHU/
├── lrw/
│
├── saved_models/
│   ├── advanced_training_report.png
│   ├── bilstm_training_history.json
│   ├── cnn_bilstm.pth
│   ├── cnn_gru.pth
│   ├── comparison_report.txt
│   ├── detailed_training_analysis.txt
│   ├── gru_training_history.json
│   ├── labels.json
│   ├── model_comparison.png
│   └── wer_cer_results.json
│
├── src/
│   ├── datasets/
│   │   ├── __pycache__/
│   │   └── lrw_dataset.py
│   │
│   └── models/
│       ├── __pycache__/
│       ├── cnn_backbone.py
│       ├── cnn_bilstm.py
│       └── cnn_gru.py
│
├── venv/
│
├── .gitignore
├── advanced_report.py
├── analyze_models.py
├── analyze_tradeoff.py
├── calculate_wer_cer.py
├── demo_video_file.py
├── diagnose_problem.py
├── README.md
├── realtime_lipreading.py
├── requirements.txt
├── setup_check.sh
├── shape_predictor_68_face_landmarks.dat
└── train.py

```

**Key Points:**
- ✅ `train.py` - Use this for training
- ✅ `demo_video_file.py` - Use this for testing
- ✅ `analyze_models.py` - Use this for evaluation

---

## 📚 Usage Examples

### **Example 1: Train Both Models**

```bash
# Train BiLSTM
python train.py --model bilstm --epochs 20

# Expected output:
# Epoch 1/20 | Train Acc: 35% | Val Acc: 32%
# Epoch 5/20 | Train Acc: 65% | Val Acc: 58%
# Epoch 10/20 | Train Acc: 82% | Val Acc: 76%
# Epoch 15/20 | Train Acc: 91% | Val Acc: 85%
# Epoch 20/20 | Train Acc: 96% | Val Acc: 88%

# Train GRU
python train.py --model gru --epochs 20
```

### **Example 2: Real-time Demonstration**

```bash
# Start demo
python realtime_lipreading.py --model bilstm

# Speak one of: PUBLIC, GREAT, PLANS, YOUNG, KILLED
# System shows:
# Prediction: GREAT (87.3%)
# Top 3: GREAT(87.3%) | PUBLIC(8.2%) | PLANS(3.1%)
# Buffer: 15/15 | FPS: 28.3 | Device: CUDA

# Press 's' to show statistics
# Press 'r' to reset statistics
# Press 'q' to quit
```

### **Example 3: Model Comparison**

```bash
# Run analysis
python analyze_models.py

# Generates:
# - Console output with detailed comparison
# - saved_models/comparison_report.txt
# - saved_models/model_comparison.png

# Output includes:
# BiLSTM: 88.7% accuracy, 45ms latency, 15.2M parameters
# GRU:    86.2% accuracy, 38ms latency, 12.8M parameters
# Trade-off: BiLSTM +2.5% accuracy, GRU 15% faster
```

### **Example 4: Diagnose Issues**

```bash
# Check system health
python diagnose_problem.py

# Outputs:
# Model weights are valid
# Dataset is reasonably balanced
# Training history shows improvement
# Model shows good output diversity
```

---

##  Common Commands Reference

```bash
# Setup
bash setup_check.sh                          # Verify environment
pip install -r requirements.txt              # Install dependencies

# Training
python train.py --model bilstm     # Train BiLSTM (20 epochs)
python train.py --model gru        # Train GRU (20 epochs)

# Inference
python demo_system.py --model bilstm        # Advanced demo
python demo_system.py --model gru        # Advanced demo
# Analysis
python analyze_models.py                     # Compare models
python analyze_tradeoff.py                   # Trade-off analysis
python compare_models.py                     # Simple comparison
python diagnose_problem.py                   # Diagnose issues
```

---

## 📖 References & Citations

### **Architecture & Methods**

1. **LipNet** - End-to-end sentence-level lip reading with spatiotemporal convolutions and Bi-LSTM
2. **ResNet-18** - Deep residual networks for visual feature extraction
3. **3D CNNs** - Spatiotemporal feature extraction from video
4. **Bi-LSTM** - Bidirectional context modeling for sequences
5. **GRU** - Efficient gated recurrent processing
6. **Transformers** - Self-attention mechanisms for sequence modeling

### **Datasets**

- LRW (Lip Reading in the Wild) - Word-level lip reading in unconstrained conditions

---

---

