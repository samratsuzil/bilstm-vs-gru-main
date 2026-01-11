"""
Comprehensive Model Evaluation Script
Evaluates trained lip reading models with all dissertation metrics:
- Top-1, Top-3, Top-5 Accuracy
- Word Error Rate (WER)
- Character Error Rate (CER)
- Per-class Performance (Precision, Recall, F1-Score)
- Confusion Matrix
"""

import torch
import json
import argparse
import os
from torch.utils.data import DataLoader
from src.datasets.lrw_dataset import LRWDataset
from src.models.cnn_bilstm import CNNBiLSTM
from src.models.cnn_gru import CNNGRU
from src.utils.metrics import calculate_all_metrics, print_metrics_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_model(model_type='bilstm', split='test', save_results=True):
    """
    Comprehensive evaluation of trained model

    Args:
        model_type: 'bilstm' or 'gru'
        split: 'val' or 'test'
        save_results: whether to save results to JSON
    """
    DATA_DIR = "lrw"
    SAVE_DIR = "saved_models"
    FRAME_SAMPLE = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Model Type: {model_type.upper()}")
    print(f"Evaluation Split: {split.upper()}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Load labels
    labels_path = os.path.join(SAVE_DIR, "labels.json")
    with open(labels_path, "r") as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}
    num_classes = len(word2idx)

    print(f"Classes ({num_classes}): {list(word2idx.keys())}\n")

    # Load dataset
    dataset = LRWDataset(DATA_DIR, split=split, num_frames=FRAME_SAMPLE, augment=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    print(f"Dataset: {len(dataset)} samples\n")

    # Load model
    if model_type.lower() == 'bilstm':
        model = CNNBiLSTM(num_classes).to(device)
        model_path = os.path.join(SAVE_DIR, "cnn_bilstm.pth")
    else:
        model = CNNGRU(num_classes).to(device)
        model_path = os.path.join(SAVE_DIR, "cnn_gru.pth")

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print(f"   Please train the model first using: python train.py --model {model_type}")
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Loaded model from {model_path}\n")

    # Calculate all metrics
    print("Calculating metrics...")
    metrics = calculate_all_metrics(model, dataloader, device, idx2word, num_classes)

    # Print detailed report
    print_metrics_report(metrics, idx2word)

    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], list(word2idx.keys()),
                          model_type, split, SAVE_DIR)

    # Save results
    if save_results:
        results_path = os.path.join(SAVE_DIR, f"{model_type}_{split}_evaluation.json")

        # Convert numpy arrays to lists for JSON serialization
        save_metrics = {
            'model_type': model_type,
            'split': split,
            'top1_accuracy': metrics['top1_accuracy'],
            'top3_accuracy': metrics['top3_accuracy'],
            'top5_accuracy': metrics['top5_accuracy'],
            'wer': metrics['wer'],
            'cer': metrics['cer'],
            'wer_details': metrics['wer_details'],
            'cer_details': metrics['cer_details'],
            'per_class_metrics': {idx2word[int(k)]: v for k, v in metrics['per_class_metrics'].items()},
            'confusion_matrix': metrics['confusion_matrix']
        }

        with open(results_path, 'w') as f:
            json.dump(save_metrics, f, indent=2)

        print(f"✓ Saved evaluation results to {results_path}\n")

    return metrics


def plot_confusion_matrix(cm, class_names, model_type, split, save_dir):
    """
    Plot and save confusion matrix
    """
    cm_array = np.array(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_type.upper()} ({split.upper()})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{model_type}_{split}_confusion_matrix.png')
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved confusion matrix to {save_path}")
    plt.close()


def compare_models(split='test'):
    """
    Compare BiLSTM and GRU models side-by-side
    """
    print(f"\n{'='*70}")
    print(f"COMPARATIVE EVALUATION: BiLSTM vs GRU")
    print(f"{'='*70}\n")

    bilstm_metrics = evaluate_model('bilstm', split, save_results=True)
    print("\n" + "="*70 + "\n")
    gru_metrics = evaluate_model('gru', split, save_results=True)

    if bilstm_metrics and gru_metrics:
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"\n{'Metric':<25} {'BiLSTM':>15} {'GRU':>15} {'Difference':>15}")
        print("-" * 70)

        metrics_to_compare = [
            ('Top-1 Accuracy (%)', 'top1_accuracy'),
            ('Top-3 Accuracy (%)', 'top3_accuracy'),
            ('Top-5 Accuracy (%)', 'top5_accuracy'),
            ('Word Error Rate (%)', 'wer'),
            ('Character Error Rate (%)', 'cer')
        ]

        for metric_name, metric_key in metrics_to_compare:
            bilstm_val = bilstm_metrics[metric_key]
            gru_val = gru_metrics[metric_key]
            diff = bilstm_val - gru_val

            print(f"{metric_name:<25} {bilstm_val:>15.2f} {gru_val:>15.2f} {diff:>+15.2f}")

        print("=" * 70)

        # Determine winner
        if bilstm_metrics['top1_accuracy'] > gru_metrics['top1_accuracy']:
            winner = "BiLSTM"
            margin = bilstm_metrics['top1_accuracy'] - gru_metrics['top1_accuracy']
        else:
            winner = "GRU"
            margin = gru_metrics['top1_accuracy'] - bilstm_metrics['top1_accuracy']

        print(f"\n🏆 Best Model (Top-1 Accuracy): {winner} (+{margin:.2f}%)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Lip Reading Model")
    parser.add_argument("--model", type=str, default=None, choices=["bilstm", "gru"],
                        help="Model type: bilstm or gru (if None, compares both)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--compare", action="store_true",
                        help="Compare both models")

    args = parser.parse_args()

    if args.compare or args.model is None:
        compare_models(args.split)
    else:
        evaluate_model(args.model, args.split, save_results=True)
