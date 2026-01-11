import torch
import json
from torch.utils.data import DataLoader
from src.datasets.lrw_dataset import LRWDataset
from src.models.cnn_bilstm import CNNBiLSTM
from src.models.cnn_gru import CNNGRU
import numpy as np


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two sequences"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate
    WER = (Substitutions + Deletions + Insertions) / Total Words
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    distance = levenshtein_distance(ref_words, hyp_words)
    wer = distance / len(ref_words) if len(ref_words) > 0 else 0
    
    return wer * 100


def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate
    CER = (Substitutions + Deletions + Insertions) / Total Characters
    """
    distance = levenshtein_distance(list(reference), list(hypothesis))
    cer = distance / len(reference) if len(reference) > 0 else 0
    
    return cer * 100


def evaluate_wer_cer(model_type='bilstm'):
    """
    Evaluate WER and CER for the model
    
    Note: Since this is a word classification task (not sequence-to-sequence),
    WER and CER are calculated on single-word predictions. For actual sequence
    generation tasks, these metrics would be more meaningful.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("saved_models/labels.json", "r") as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}
    num_classes = len(word2idx)
    
    if model_type.lower() == 'bilstm':
        model = CNNBiLSTM(num_classes).to(device)
        model.load_state_dict(torch.load("saved_models/cnn_bilstm.pth", 
                                         map_location=device))
    else:
        model = CNNGRU(num_classes).to(device)
        model.load_state_dict(torch.load("saved_models/cnn_gru.pth", 
                                         map_location=device))
    
    model.eval()
    
    test_dataset = LRWDataset("lrw", split="test", num_frames=15)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    wer_scores = []
    cer_scores = []
    
    print(f"\nEvaluating {model_type.upper()} model...")
    print("="*60)
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = outputs.max(1)
            
            for pred, true in zip(preds, y):
                pred_word = idx2word[pred.item()]
                true_word = idx2word[true.item()]
                
                wer = calculate_wer(true_word, pred_word)
                cer = calculate_cer(true_word, pred_word)
                
                wer_scores.append(wer)
                cer_scores.append(cer)
    
    avg_wer = np.mean(wer_scores)
    avg_cer = np.mean(cer_scores)
    
    correct_predictions = sum(1 for w in wer_scores if w == 0)
    accuracy = (correct_predictions / len(wer_scores)) * 100
    
    print(f"\nResults for {model_type.upper()}:")
    print(f"  Word Error Rate (WER): {avg_wer:.2f}%")
    print(f"  Character Error Rate (CER): {avg_cer:.2f}%")
    print(f"  Accuracy (for comparison): {accuracy:.2f}%")
    print(f"  Total predictions: {len(wer_scores)}")
    
    print("\n" + "="*60)
    print("IMPORTANT NOTE:")
    print("="*60)
    print("WER and CER are typically used for sequence-to-sequence tasks")
    print("(e.g., speech-to-text transcription). For word classification,")
    print("accuracy is the more appropriate metric. These metrics are")
    print("included for proposal completeness.")
    print("="*60)
    
    return {
        'model': model_type,
        'wer': avg_wer,
        'cer': avg_cer,
        'accuracy': accuracy,
        'total_samples': len(wer_scores)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate WER and CER metrics")
    parser.add_argument("--model", type=str, default="both",
                        choices=["bilstm", "gru", "both"],
                        help="Which model to evaluate")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Word Error Rate (WER) & Character Error Rate (CER) Evaluation")
    print("="*60)
    
    results = []
    
    if args.model == "both":
        results.append(evaluate_wer_cer("bilstm"))
        results.append(evaluate_wer_cer("gru"))
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        for r in results:
            print(f"\n{r['model'].upper()}:")
            print(f"  WER: {r['wer']:.2f}%")
            print(f"  CER: {r['cer']:.2f}%")
            print(f"  Accuracy: {r['accuracy']:.2f}%")
    else:
        results.append(evaluate_wer_cer(args.model))
    
    with open("saved_models/wer_cer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to saved_models/wer_cer_results.json")
