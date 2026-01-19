import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from src.datasets.lrw_dataset import LRWDataset
from src.models.cnn_bilstm import CNNBiLSTM
from src.models.cnn_gru import CNNGRU
from src.utils.metrics import calculate_wer, calculate_cer, calculate_top_k_accuracy
import time
import jsond
import os
import argparse
from collections import defaultdict

def train_model(model_type='bilstm', epochs=20, batch_size=16, lr=1e-3,
                dropout=0.3, weight_decay=1e-4, early_stopping=True, patience=15,
                augment=True, use_resnet50=False, hidden_size=512, num_layers=2):
    """
    Train lip reading model with enhanced configuration for 85%+ accuracy

    Args:
        model_type: 'bilstm' or 'gru'
        epochs: number of training epochs (recommended: 150-200 for 85%+ accuracy)
        batch_size: batch size for training (default: 16 - increased for stability)
        lr: learning rate (recommended: 1e-3 with cosine annealing)
        dropout: dropout rate for regularization (default: 0.3 - reduced overfitting)
        weight_decay: L2 regularization weight (default: 1e-4)
        early_stopping: whether to use early stopping (default: True)
        patience: early stopping patience in epochs (default: 15 - increased)
        augment: whether to use aggressive data augmentation (default: True)
        use_resnet50: use ResNet50 instead of ResNet18 for better features (default: False)
        hidden_size: RNN hidden size - larger is better but slower (default: 512)
        num_layers: number of RNN layers - 2 is optimal for speed/accuracy (default: 2)
    """
    DATA_DIR = "lrw"
    SAVE_DIR = "saved_models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    FRAME_SAMPLE = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: {model_type.upper()}")
    print(f"  CNN Backbone: {'ResNet50' if use_resnet50 else 'ResNet18'}")
    print(f"  RNN Hidden Size: {hidden_size}")
    print(f"  RNN Layers: {num_layers}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Dropout: {dropout}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Early Stopping: {early_stopping} (patience={patience})")
    print(f"  Aggressive Augmentation: {augment}")
    print(f"  Frame Sampling: {FRAME_SAMPLE}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    import torchvision.transforms as transforms
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(112, scale=(0.9, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(p=0.3)
        ])
        print("Data augmentation enabled for training set\n")
    else:
        train_transform = None
    
    train_ds = LRWDataset(DATA_DIR, split="train", num_frames=FRAME_SAMPLE, transform=train_transform, augment=augment)
    val_ds = LRWDataset(DATA_DIR, split="val", num_frames=FRAME_SAMPLE, augment=False)
    test_ds = LRWDataset(DATA_DIR, split="test", num_frames=FRAME_SAMPLE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    print(f"Dataset Statistics:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples: {len(val_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    print(f"  Classes: {train_ds.word2idx}")
    
    class_counts = defaultdict(int)
    for _, label, _ in train_ds.samples:
        class_counts[label] += 1
    
    print(f"\nClass Distribution in Training Set:")
    for word, idx in sorted(train_ds.word2idx.items(), key=lambda x: x[1]):
        count = class_counts[idx]
        percentage = 100 * count / len(train_ds)
        print(f"  {word}: {count} samples ({percentage:.1f}%)")
    print()

    with open(os.path.join(SAVE_DIR, "labels.json"), "w") as f:
        json.dump(train_ds.word2idx, f, indent=2)
    print("Saved labels.json\n")

    num_classes = len(train_ds.word2idx)
    if model_type.lower() == 'bilstm':
        model = CNNBiLSTM(num_classes, dropout=dropout, use_resnet50=use_resnet50,
                         hidden_size=hidden_size, num_layers=num_layers).to(device)
        save_path = os.path.join(SAVE_DIR, "cnn_bilstm.pth")
    else:
        model = CNNGRU(num_classes, dropout=dropout, use_resnet50=use_resnet50,
                      hidden_size=hidden_size, num_layers=num_layers).to(device)
        save_path = os.path.join(SAVE_DIR, "cnn_gru.pth")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}\n")

    # Calculate class weights for balanced training
    class_samples = [class_counts[idx] for idx in range(num_classes)]
    total_samples = sum(class_samples)
    class_weights = torch.tensor([total_samples / (num_classes * count) for count in class_samples],
                                  dtype=torch.float32).to(device)

    print(f"Class Weights for Balanced Loss:")
    for word, idx in sorted(train_ds.word2idx.items(), key=lambda x: x[1]):
        print(f"  {word}: {class_weights[idx]:.4f}")
    print()

    # Use label smoothing to prevent overconfident predictions
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # AdamW optimizer with better weight decay handling
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    # Cosine annealing with warm restarts for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    idx2word = {v: k for k, v in train_ds.word2idx.items()}

    best_val_acc = 0
    best_val_wer = 100.0
    patience_counter = 0
    training_history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': [],
        'val_wer': [],
        'val_cer': [],
        'val_top3_acc': [],
        'val_top5_acc': []
    }

    # Mixed precision training for faster inference and better memory usage
    use_amp = torch.cuda.is_available()
    scaler = GradScaler('cuda') if use_amp else None

    if use_amp:
        print("Mixed Precision Training (AMP) enabled for faster training and inference\n")

    total_training_start = time.time()

    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")

        model.train()
        train_correct = train_total = 0
        train_loss_sum = 0
        start_time = time.time()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if use_amp:
                with autocast('cuda'):
                    outputs = model(x)
                    loss = criterion(outputs, y)

                # Scaled backward pass
                scaler.scale(loss).backward()

                # Gradient clipping with scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            _, preds = outputs.max(1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            train_loss_sum += loss.item()

            if i % 10 == 0:
                batch_acc = 100 * (preds == y).sum().item() / y.size(0)
                print(f"  Batch {i:3d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Batch Acc: {batch_acc:.2f}%")

        train_acc = 100 * train_correct / train_total
        train_loss = train_loss_sum / len(train_loader)
        train_time = time.time() - start_time

        model.eval()
        val_correct = val_total = 0
        val_loss_sum = 0
        val_start = time.time()

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        all_preds = []
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)

                _, preds = outputs.max(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                val_loss_sum += loss.item()

                # Collect for metrics
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())
                all_outputs.append(outputs.cpu())

                for pred, true in zip(preds, y):
                    class_total[true.item()] += 1
                    if pred == true:
                        class_correct[true.item()] += 1

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)
        val_time = time.time() - val_start

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.tensor(all_labels)

        val_wer, wer_details = calculate_wer(all_preds, all_labels, idx2word)
        val_cer, cer_details = calculate_cer(all_preds, all_labels, idx2word)
        val_top3 = calculate_top_k_accuracy(all_outputs, all_targets, k=min(3, num_classes))
        val_top5 = calculate_top_k_accuracy(all_outputs, all_targets, k=min(5, num_classes))

        # Step the cosine annealing scheduler
        scheduler.step()

        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_wer'].append(val_wer)
        training_history['val_cer'].append(val_cer)
        training_history['val_top3_acc'].append(val_top3)
        training_history['val_top5_acc'].append(val_top5)

        # Calculate how many classes are actually being predicted
        num_classes_predicted = sum(1 for idx in range(num_classes) if class_correct.get(idx, 0) > 0)

        total_time = train_time + val_time

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs} Summary (Time: {total_time:.1f}s)")
        print(f"{'='*80}")
        print(f"Train:      Acc={train_acc:.2f}%  Loss={train_loss:.4f}  Time={train_time:.1f}s")
        print(f"Validation: Acc={val_acc:.2f}%  Loss={val_loss:.4f}  Time={val_time:.1f}s")
        print(f"{'-'*80}")
        print(f"Top-3 Acc: {val_top3:.2f}%  |  Top-5 Acc: {val_top5:.2f}%")
        print(f"WER: {val_wer:.2f}%  |  CER: {val_cer:.2f}%")
        print(f"Classes Predicted: {num_classes_predicted}/{num_classes}")
        print(f"{'='*80}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_wer = val_wer
            torch.save(model.state_dict(), save_path)
            print(f"\nPASS Best model saved! Val Acc: {best_val_acc:.2f}% | WER: {best_val_wer:.2f}%")
            patience_counter = 0
        else:
            if early_stopping:
                patience_counter += 1
                if patience_counter >= patience:
                    total_training_time = time.time() - total_training_start
                    print(f"\n🛑 Early stopping at epoch {epoch+1} (Best Val Acc: {best_val_acc:.2f}%)")
                    print(f"Total Training Time: {total_training_time/60:.1f} minutes")
                    break

    total_training_time = time.time() - total_training_start
    print(f"\n{'='*80}")
    print(f"Training Complete! Total Time: {total_training_time/60:.1f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*80}")

    print(f"\n{'='*80}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*80}")

    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_correct = test_total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    all_test_preds = []
    all_test_labels = []
    all_test_outputs = []

    # Measure inference time for real-time assessment
    inference_times = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # Measure inference time per batch
            inf_start = time.time()
            outputs = model(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inf_time = time.time() - inf_start
            inference_times.append(inf_time / x.size(0))  # per sample

            _, preds = outputs.max(1)

            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

            all_test_preds.extend(preds.cpu().numpy().tolist())
            all_test_labels.extend(y.cpu().numpy().tolist())
            all_test_outputs.append(outputs.cpu())

            for pred, true in zip(preds, y):
                class_total[true.item()] += 1
                if pred == true:
                    class_correct[true.item()] += 1

    test_acc = 100 * test_correct / test_total

    all_test_outputs = torch.cat(all_test_outputs, dim=0)
    all_test_targets = torch.tensor(all_test_labels)

    test_wer, test_wer_details = calculate_wer(all_test_preds, all_test_labels, idx2word)
    test_cer, test_cer_details = calculate_cer(all_test_preds, all_test_labels, idx2word)
    test_top3 = calculate_top_k_accuracy(all_test_outputs, all_test_targets, k=min(3, num_classes))
    test_top5 = calculate_top_k_accuracy(all_test_outputs, all_test_targets, k=min(5, num_classes))

    test_classes_predicted = sum(1 for idx in range(num_classes) if class_correct.get(idx, 0) > 0)

    # Calculate average inference time and FPS
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    print(f"\n{'='*80}")
    print(f"FINAL TEST RESULTS - {model_type.upper()} MODEL")
    print(f"{'='*80}")
    print(f"\n--- Recognition Accuracy (Research Objective: 85-95%) ---")
    print(f"Top-1 Accuracy: {test_acc:.2f}%")
    print(f"Top-3 Accuracy: {test_top3:.2f}%")
    print(f"Top-5 Accuracy: {test_top5:.2f}%")
    print(f"{'-'*80}")
    print(f"\n--- Error Rates ---")
    print(f"Word Error Rate (WER): {test_wer:.2f}%")
    print(f"Character Error Rate (CER): {test_cer:.2f}%")
    print(f"{'-'*80}")
    print(f"\n--- Detailed Statistics ---")
    print(f"Total Words: {test_wer_details['total_words']} | Correct: {test_wer_details['correct']} | Errors: {test_wer_details['errors']}")
    print(f"Total Characters: {test_cer_details['total_characters']} | Errors: {test_cer_details['character_errors']}")
    print(f"Classes Predicted: {test_classes_predicted}/{num_classes}")
    print(f"{'-'*80}")
    print(f"\n--- Real-Time Performance (Research Objective: <200ms latency) ---")
    print(f"Average Inference Time: {avg_inference_time*1000:.2f} ms per sample")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print(f"Real-time Capable (30 FPS): {'YES PASS' if fps >= 30 else 'NO FAIL'}")
    print(f"Low Latency (<200ms): {'YES PASS' if avg_inference_time*1000 < 200 else 'NO FAIL'}")
    print(f"{'-'*80}")
    print(f"\n--- Model Complexity ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (float32)")
    print(f"Total Training Time: {total_training_time/60:.1f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*80}")

    # Determine if research objectives are met
    objectives_met = {
        'accuracy_goal': test_acc >= 85.0,
        'real_time_latency': avg_inference_time*1000 < 200,
        'real_time_fps': fps >= 30
    }

    print(f"\n{'='*80}")
    print("RESEARCH OBJECTIVES ASSESSMENT")
    print(f"{'='*80}")
    print(f"Accuracy Target (85-95%): {'ACHIEVED PASS' if objectives_met['accuracy_goal'] else 'NOT MET FAIL'} ({test_acc:.2f}%)")
    print(f"Latency Target (<200ms): {'ACHIEVED PASS' if objectives_met['real_time_latency'] else 'NOT MET FAIL'} ({avg_inference_time*1000:.2f}ms)")
    print(f"FPS Target (>=30 FPS): {'ACHIEVED PASS' if objectives_met['real_time_fps'] else 'NOT MET FAIL'} ({fps:.2f} FPS)")
    print(f"Overall: {'ALL OBJECTIVES MET PASSPASSPASS' if all(objectives_met.values()) else 'SOME OBJECTIVES PENDING'}")
    print(f"{'='*80}\n")

    training_history['test_acc'] = test_acc
    training_history['test_wer'] = test_wer
    training_history['test_cer'] = test_cer
    training_history['test_top3_acc'] = test_top3
    training_history['test_top5_acc'] = test_top5
    training_history['avg_inference_time_ms'] = avg_inference_time * 1000
    training_history['fps'] = fps
    training_history['total_training_time_minutes'] = total_training_time / 60
    training_history['objectives_met'] = objectives_met
    training_history['total_parameters'] = total_params
    training_history['model_size_mb'] = total_params * 4 / (1024**2)

    history_path = os.path.join(SAVE_DIR, f"{model_type}_training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Save comprehensive final report
    report_path = os.path.join(SAVE_DIR, f"{model_type}_final_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"COMPARATIVE STUDY: {model_type.upper()} MODEL FOR LIP READING\n")
        f.write("="*80 + "\n\n")
        f.write("RESEARCH CONTEXT\n")
        f.write("-"*80 + "\n")
        f.write("This model is part of a comparative study of Bi-LSTM and GRU architectures\n")
        f.write("for real-time visual-only speech recognition (VSR/Lip Reading).\n")
        f.write("Dataset: Lip Reading in the Wild (LRW)\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Architecture: CNN + {model_type.upper()}\n")
        f.write(f"CNN Backbone: {'ResNet50' if use_resnet50 else 'ResNet18'}\n")
        f.write(f"RNN Hidden Size: {hidden_size}\n")
        f.write(f"RNN Layers: {num_layers}\n")
        f.write(f"Dropout: {dropout}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Model Size: {total_params * 4 / (1024**2):.2f} MB\n\n")

        f.write("TRAINING CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Data Augmentation: {'Enabled' if augment else 'Disabled'}\n")
        f.write(f"Total Training Time: {total_training_time/60:.1f} minutes\n\n")

        f.write("RECOGNITION ACCURACY RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Top-1 Accuracy: {test_acc:.2f}% (Target: 85-95%)\n")
        f.write(f"Top-3 Accuracy: {test_top3:.2f}%\n")
        f.write(f"Top-5 Accuracy: {test_top5:.2f}%\n")
        f.write(f"Word Error Rate (WER): {test_wer:.2f}%\n")
        f.write(f"Character Error Rate (CER): {test_cer:.2f}%\n\n")

        f.write("REAL-TIME PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Avg Inference Time: {avg_inference_time*1000:.2f} ms (Target: <200ms)\n")
        f.write(f"Frames Per Second: {fps:.2f} FPS (Target: >=30 FPS)\n")
        f.write(f"Real-time Capable: {'YES' if fps >= 30 else 'NO'}\n\n")

        f.write("RESEARCH OBJECTIVES STATUS\n")
        f.write("-"*80 + "\n")
        f.write(f"[{'PASS' if objectives_met['accuracy_goal'] else 'FAIL'}] Accuracy Goal (85-95%): {test_acc:.2f}%\n")
        f.write(f"[{'PASS' if objectives_met['real_time_latency'] else 'FAIL'}] Latency Target (<200ms): {avg_inference_time*1000:.2f}ms\n")
        f.write(f"[{'PASS' if objectives_met['real_time_fps'] else 'FAIL'}] FPS Target (>=30): {fps:.2f}\n")
        f.write(f"\nOverall: {'ALL OBJECTIVES ACHIEVED' if all(objectives_met.values()) else 'PARTIAL SUCCESS'}\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"Saved comprehensive report to {report_path}\n")

    return model, training_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lip Reading Model")
    parser.add_argument("--model", type=str, default="bilstm", choices=["bilstm", "gru"],
                        help="Model type: bilstm or gru")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs (default: 150 for 85%+ accuracy)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16 for better gradient estimates)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001 with cosine annealing)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3 - reduced for better learning)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for L2 regularization (default: 0.0001)")
    parser.add_argument("--early_stopping", action="store_true", default=True,
                        help="Enable early stopping (default: True)")
    parser.add_argument("--no_early_stopping", action="store_false", dest="early_stopping",
                        help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15 - increased for better convergence)")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable aggressive data augmentation (default: True)")
    parser.add_argument("--no_augment", action="store_false", dest="augment",
                        help="Disable data augmentation")
    parser.add_argument("--resnet50", action="store_true", default=False,
                        help="Use ResNet50 instead of ResNet18 for better features (slower, more VRAM)")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="RNN hidden size (default: 512, larger = better but slower)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of RNN layers (default: 2 for optimal speed/accuracy)")

    args = parser.parse_args()

    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        patience=args.patience,
        augment=args.augment,
        use_resnet50=args.resnet50,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )