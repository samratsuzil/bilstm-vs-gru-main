#!/bin/bash

##############################################################################
# Lip Reading Training Script - Comparative Study of Bi-LSTM vs GRU
# Based on Research Proposal: A Comparative Study of Bi-LSTM and GRU
# Architectures for Lip Reading
#
# This script trains both models and generates a comprehensive comparison
# report to satisfy the research objectives.
##############################################################################

echo "=========================================================================="
echo "COMPARATIVE STUDY: Bi-LSTM vs GRU for Lip Reading"
echo "=========================================================================="
echo ""
echo "Research Objectives:"
echo "  - Develop CNN+Bi-LSTM and CNN+GRU models"
echo "  - Compare accuracy, training time, inference latency, model size"
echo "  - Target: 85-95% accuracy with real-time capability"
echo ""
echo "Dataset: LRW (Lip Reading in the Wild)"
echo "=========================================================================="
echo ""

# Configuration (modify these as needed)
EPOCHS=100
BATCH_SIZE=8
LEARNING_RATE=0.00005
DROPOUT=0.5
WEIGHT_DECAY=0.0001
HIDDEN_SIZE=512
NUM_LAYERS=3
USE_RESNET50=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --resnet50)
      USE_RESNET50=true
      shift
      ;;
    --quick)
      EPOCHS=20
      echo "Quick training mode: $EPOCHS epochs"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--epochs N] [--batch_size N] [--resnet50] [--quick]"
      exit 1
      ;;
  esac
done

echo "Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Dropout: $DROPOUT"
echo "  RNN Hidden Size: $HIDDEN_SIZE"
echo "  RNN Layers: $NUM_LAYERS"
echo "  CNN Backbone: $([ "$USE_RESNET50" = true ] && echo 'ResNet50' || echo 'ResNet18')"
echo ""

# Check if dataset exists
if [ ! -d "lrw" ]; then
    echo "ERROR: LRW dataset not found in ./lrw directory"
    echo "Please ensure the dataset is properly organized in the lrw/ folder"
    exit 1
fi

echo "Dataset found. Proceeding with training..."
echo ""

# Train Bi-LSTM model
echo "=========================================================================="
echo "PHASE 1: Training CNN + Bi-LSTM Model"
echo "=========================================================================="
echo ""

python train.py \
    --model bilstm \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --dropout $DROPOUT \
    --weight_decay $WEIGHT_DECAY \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    $([ "$USE_RESNET50" = true ] && echo "--resnet50") \
    --augment

if [ $? -ne 0 ]; then
    echo "ERROR: Bi-LSTM training failed"
    exit 1
fi

echo ""
echo "Bi-LSTM training completed successfully!"
echo ""

# Train GRU model
echo "=========================================================================="
echo "PHASE 2: Training CNN + GRU Model"
echo "=========================================================================="
echo ""

python train.py \
    --model gru \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --dropout $DROPOUT \
    --weight_decay $WEIGHT_DECAY \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    $([ "$USE_RESNET50" = true ] && echo "--resnet50") \
    --augment

if [ $? -ne 0 ]; then
    echo "ERROR: GRU training failed"
    exit 1
fi

echo ""
echo "GRU training completed successfully!"
echo ""

# Generate comparative analysis
echo "=========================================================================="
echo "PHASE 3: Generating Comparative Analysis Report"
echo "=========================================================================="
echo ""

python compare_models.py

if [ $? -ne 0 ]; then
    echo "WARNING: Comparison report generation failed, but models are trained"
fi

echo ""
echo "=========================================================================="
echo "TRAINING COMPLETE"
echo "=========================================================================="
echo ""
echo "Results saved in:"
echo "  - saved_models/bilstm_final_report.txt"
echo "  - saved_models/gru_final_report.txt"
echo "  - saved_models/bilstm_training_history.json"
echo "  - saved_models/gru_training_history.json"
echo "  - saved_models/comparative_analysis.txt (if generated)"
echo ""
echo "Models saved in:"
echo "  - saved_models/cnn_bilstm.pth"
echo "  - saved_models/cnn_gru.pth"
echo ""
echo "=========================================================================="
