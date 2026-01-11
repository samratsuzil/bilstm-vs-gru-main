import os
import json
import torch
import numpy as np
from collections import defaultdict


def check_model_weights(model_path, model_name):
    """Check if model weights are properly initialized"""
    print(f"\nChecking {model_name} model weights...")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    
    weights = torch.load(model_path, map_location='cpu')
    
    has_nan = False
    has_inf = False
    total_params = 0
    
    for key, tensor in weights.items():
        total_params += tensor.numel()
        if torch.isnan(tensor).any():
            has_nan = True
            print(f"  Warning: NaN values in {key}")
        if torch.isinf(tensor).any():
            has_inf = True
            print(f"  Warning: Inf values in {key}")
    
    if not has_nan and not has_inf:
        print(f"Weights are valid")
        print(f"Total parameters: {total_params:,}")
    
    fc_keys = [k for k in weights.keys() if 'fc' in k.lower() or 'classifier' in k.lower()]
    if fc_keys:
        print(f"\n  Classifier layer analysis:")
        for key in fc_keys:
            tensor = weights[key]
            print(f"    {key}: shape={tensor.shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")
    
    return True


def check_dataset_balance():
    """Check if dataset is balanced"""
    print("\n" + "="*70)
    print("DATASET BALANCE CHECK")
    print("="*70)
    
    data_dir = "lrw"
    if not os.path.exists(data_dir):
        print("Dataset directory not found")
        return
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n{split.upper()} Split:")
        class_counts = defaultdict(int)
        
        for word in os.listdir(data_dir):
            word_path = os.path.join(data_dir, word, split)
            if os.path.isdir(word_path):
                num_files = len([f for f in os.listdir(word_path) if f.endswith('.mp4') or f.endswith('.txt')])
                class_counts[word] = num_files
        
        if not class_counts:
            print("  No data found")
            continue
        
        total = sum(class_counts.values())
        
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"  Total samples: {total}")
        for word, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / total if total > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"    {word:12s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        if imbalance_ratio > 2:
            print(f"\nWARNING: Dataset is imbalanced (ratio {imbalance_ratio:.1f}:1)")
            print(f"Most common class has {max_count} samples")
            print(f"Least common class has {min_count} samples")
            print(f"This can cause the model to always predict the majority class!")
        else:
            print(f"\nDataset is reasonably balanced")


def check_training_history():
    """Check training history if available"""
    print("\n" + "="*70)
    print("TRAINING HISTORY CHECK")
    print("="*70)
    
    for model_type in ['bilstm', 'gru']:
        history_path = f"saved_models/{model_type}_training_history.json"
        
        if os.path.exists(history_path):
            print(f"\n{model_type.upper()} Training History:")
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            if 'train_acc' in history and history['train_acc']:
                print(f"  Epochs trained: {len(history['train_acc'])}")
                print(f"  Final train accuracy: {history['train_acc'][-1]:.2f}%")
                print(f"  Final val accuracy: {history['val_acc'][-1]:.2f}%")
                
                # Check for overfitting
                if history['train_acc'][-1] - history['val_acc'][-1] > 20:
                    print(f"  WARNING: Large gap between train and val accuracy (overfitting)")
                
                # Check if model improved
                if len(history['val_acc']) > 1:
                    improvement = history['val_acc'][-1] - history['val_acc'][0]
                    if improvement < 5:
                        print(f"  WARNING: Model barely improved ({improvement:.2f}% gain)")
        else:
            print(f"\n{model_type.upper()}: No training history found")


def test_model_output_diversity():
    """Test if model produces diverse outputs"""
    print("\n" + "="*70)
    print("MODEL OUTPUT DIVERSITY CHECK")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("saved_models/labels.json", "r") as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}
    num_classes = len(word2idx)
    
    for model_type in ['bilstm', 'gru']:
        model_path = f"saved_models/cnn_{model_type}.pth"
        
        if not os.path.exists(model_path):
            continue
        
        print(f"\n{model_type.upper()} Model:")
        
        if model_type == 'bilstm':
            from src.models.cnn_bilstm import CNNBiLSTM
            model = CNNBiLSTM(num_classes).to(device)
        else:
            from src.models.cnn_gru import CNNGRU
            model = CNNGRU(num_classes).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        predictions = []
        confidences = []
        
        for _ in range(20):
            x = torch.randn(1, 15, 1, 112, 112).to(device)
            
            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)
                conf, pred = probs.max(1)
                
                predictions.append(pred.item())
                confidences.append(conf.item())
        
        unique_predictions = len(set(predictions))
        most_common = max(set(predictions), key=predictions.count)
        most_common_count = predictions.count(most_common)
        
        print(f"  Random input tests: 20")
        print(f"  Unique predictions: {unique_predictions}/{num_classes}")
        print(f"  Most common: {idx2word[most_common]} ({most_common_count}/20 = {100*most_common_count/20:.1f}%)")
        print(f"  Avg confidence: {np.mean(confidences):.2%}")
        
        if unique_predictions == 1:
            print(f"CRITICAL: Model ALWAYS predicts '{idx2word[most_common]}'!")
            print(f"This indicates the model didn't learn properly.")
            print(f"Solution: Retrain with 'python train.py --model {model_type} --epochs 20'")
        elif unique_predictions < num_classes / 2:
            print(f"WARNING: Low output diversity")
            print(f"Model may be undertrained or biased")
        else:
            print(f"Model shows good output diversity")


def main():
    print("\n" + "="*70)
    print("LIP READING SYSTEM - PROBLEM DIAGNOSIS")
    print("="*70)
    print("\nThis tool helps identify why your model might always predict the same word")
    print()
    
    print("\n" + "="*70)
    print("MODEL WEIGHTS CHECK")
    print("="*70)
    
    check_model_weights("saved_models/cnn_bilstm.pth", "BiLSTM")
    check_model_weights("saved_models/cnn_gru.pth", "GRU")
    
    check_dataset_balance()
    
    check_training_history()
    
    test_model_output_diversity()

if __name__ == "__main__":
    main()
