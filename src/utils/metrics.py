"""
Evaluation Metrics for Lip Reading System
Includes WER, CER, Top-K Accuracy for classification-based word recognition
"""

import numpy as np
import torch
from collections import defaultdict


def calculate_wer(predictions, references, idx2word):
    """
    Calculate Word Error Rate for classification task

    Args:
        predictions: list of predicted class indices
        references: list of ground truth class indices
        idx2word: mapping from index to word

    Returns:
        wer: Word Error Rate (percentage)
        details: dictionary with breakdown
    """
    assert len(predictions) == len(references), "Predictions and references must have same length"

    total_words = len(references)
    errors = sum(1 for pred, ref in zip(predictions, references) if pred != ref)

    wer = 100.0 * errors / total_words if total_words > 0 else 0.0

    details = {
        'total_words': total_words,
        'errors': errors,
        'correct': total_words - errors,
        'wer': wer,
        'accuracy': 100.0 - wer
    }

    return wer, details


def calculate_cer(predictions, references, idx2word):
    """
    Calculate Character Error Rate for classification task
    Uses Levenshtein distance at character level

    Args:
        predictions: list of predicted class indices
        references: list of ground truth class indices
        idx2word: mapping from index to word

    Returns:
        cer: Character Error Rate (percentage)
        details: dictionary with breakdown
    """
    total_chars = 0
    total_char_errors = 0

    for pred_idx, ref_idx in zip(predictions, references):
        pred_word = idx2word[pred_idx]
        ref_word = idx2word[ref_idx]

        char_errors = levenshtein_distance(pred_word, ref_word)
        total_char_errors += char_errors
        total_chars += len(ref_word)

    cer = 100.0 * total_char_errors / total_chars if total_chars > 0 else 0.0

    details = {
        'total_characters': total_chars,
        'character_errors': total_char_errors,
        'cer': cer
    }

    return cer, details


def levenshtein_distance(s1, s2):
    """
    Calculate Levenshtein (edit) distance between two strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_top_k_accuracy(outputs, targets, k=5):
    """
    Calculate Top-K accuracy

    Args:
        outputs: model outputs (logits or probabilities) - shape (N, num_classes)
        targets: ground truth labels - shape (N,)
        k: top-k value

    Returns:
        top_k_acc: Top-K accuracy (percentage)
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        top_k_acc = correct_k.mul_(100.0 / batch_size).item()

    return top_k_acc


def calculate_per_class_metrics(predictions, references, num_classes):
    """
    Calculate per-class accuracy, precision, recall, F1-score

    Args:
        predictions: list of predicted class indices
        references: list of ground truth class indices
        num_classes: total number of classes

    Returns:
        metrics: dictionary with per-class metrics
    """
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_pred_count = defaultdict(int)

    # True positives, false positives, false negatives per class
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, ref in zip(predictions, references):
        class_total[ref] += 1
        class_pred_count[pred] += 1

        if pred == ref:
            class_correct[ref] += 1
            tp[ref] += 1
        else:
            fp[pred] += 1
            fn[ref] += 1

    metrics = {}
    for cls in range(num_classes):
        total = class_total.get(cls, 0)
        correct = class_correct.get(cls, 0)
        pred_count = class_pred_count.get(cls, 0)

        accuracy = 100.0 * correct / total if total > 0 else 0.0

        # Precision = TP / (TP + FP)
        precision = 100.0 * tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall = 100.0 * tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[cls] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': total,
            'correct_predictions': correct,
            'predicted_as_this_class': pred_count
        }

    return metrics


def calculate_confusion_matrix(predictions, references, num_classes):
    """
    Calculate confusion matrix

    Args:
        predictions: list of predicted class indices
        references: list of ground truth class indices
        num_classes: total number of classes

    Returns:
        confusion_matrix: numpy array of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for pred, ref in zip(predictions, references):
        cm[ref][pred] += 1

    return cm


def calculate_all_metrics(model, data_loader, device, idx2word, num_classes):
    """
    Calculate all metrics for a given model and dataset

    Args:
        model: trained model
        data_loader: DataLoader for evaluation
        device: torch device
        idx2word: mapping from index to word
        num_classes: total number of classes

    Returns:
        metrics_dict: dictionary containing all metrics
    """
    model.eval()

    all_predictions = []
    all_references = []
    all_outputs = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = outputs.max(1)

            all_predictions.extend(preds.cpu().numpy().tolist())
            all_references.extend(y.cpu().numpy().tolist())
            all_outputs.append(outputs.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.tensor(all_references)

    wer, wer_details = calculate_wer(all_predictions, all_references, idx2word)
    cer, cer_details = calculate_cer(all_predictions, all_references, idx2word)

    top1_acc = calculate_top_k_accuracy(all_outputs, all_targets, k=1)
    top3_acc = calculate_top_k_accuracy(all_outputs, all_targets, k=min(3, num_classes))
    top5_acc = calculate_top_k_accuracy(all_outputs, all_targets, k=min(5, num_classes))

    per_class = calculate_per_class_metrics(all_predictions, all_references, num_classes)

    cm = calculate_confusion_matrix(all_predictions, all_references, num_classes)

    metrics_dict = {
        'wer': wer,
        'wer_details': wer_details,
        'cer': cer,
        'cer_details': cer_details,
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'per_class_metrics': per_class,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions,
        'references': all_references
    }

    return metrics_dict


def print_metrics_report(metrics, idx2word):
    """
    Print a formatted metrics report

    Args:
        metrics: metrics dictionary from calculate_all_metrics
        idx2word: mapping from index to word
    """
    print("\n" + "="*70)
    print("EVALUATION METRICS REPORT")
    print("="*70)

    print("\n--- Overall Performance ---")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    print(f"Top-3 Accuracy: {metrics['top3_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    print(f"Word Error Rate (WER): {metrics['wer']:.2f}%")
    print(f"Character Error Rate (CER): {metrics['cer']:.2f}%")

    print("\n--- Word Error Rate Details ---")
    wer_det = metrics['wer_details']
    print(f"Total Words: {wer_det['total_words']}")
    print(f"Correct: {wer_det['correct']}")
    print(f"Errors: {wer_det['errors']}")

    print("\n--- Character Error Rate Details ---")
    cer_det = metrics['cer_details']
    print(f"Total Characters: {cer_det['total_characters']}")
    print(f"Character Errors: {cer_det['character_errors']}")

    print("\n--- Per-Class Performance ---")
    print(f"{'Class':<12} {'Acc%':>7} {'Prec%':>7} {'Rec%':>7} {'F1%':>7} {'Samples':>8}")
    print("-" * 70)

    per_class = metrics['per_class_metrics']
    for idx in sorted(per_class.keys()):
        word = idx2word[idx]
        pm = per_class[idx]
        print(f"{word:<12} {pm['accuracy']:>7.2f} {pm['precision']:>7.2f} "
              f"{pm['recall']:>7.2f} {pm['f1_score']:>7.2f} {pm['total_samples']:>8}")

    avg_acc = np.mean([pm['accuracy'] for pm in per_class.values()])
    avg_prec = np.mean([pm['precision'] for pm in per_class.values()])
    avg_rec = np.mean([pm['recall'] for pm in per_class.values()])
    avg_f1 = np.mean([pm['f1_score'] for pm in per_class.values()])

    print("-" * 70)
    print(f"{'Average':<12} {avg_acc:>7.2f} {avg_prec:>7.2f} "
          f"{avg_rec:>7.2f} {avg_f1:>7.2f}")

    print("\n" + "="*70 + "\n")
