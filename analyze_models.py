import torch
import json
import time
import numpy as np
from torch.utils.data import DataLoader
from src.datasets.lrw_dataset import LRWDataset
from src.models.cnn_bilstm import CNNBiLSTM
from src.models.cnn_gru import CNNGRU
from src.utils.metrics import calculate_all_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os


class ModelAnalyzer:
    """Comprehensive model comparison and analysis for dissertation"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open("saved_models/labels.json", "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.num_classes = len(self.word2idx)

        self.test_dataset = LRWDataset("lrw", split="test", num_frames=15, augment=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=8, shuffle=False)

        print(f"\n{'='*70}")
        print(f"MODEL ANALYZER INITIALIZED")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Test Samples: {len(self.test_dataset)}")
        print(f"Classes ({self.num_classes}): {list(self.word2idx.keys())}")
        print(f"{'='*70}\n")

    def load_model(self, model_type):
        """Load trained model"""
        if model_type.lower() == 'bilstm':
            model = CNNBiLSTM(self.num_classes).to(self.device)
            model.load_state_dict(torch.load("saved_models/cnn_bilstm.pth",
                                            map_location=self.device))
        else:
            model = CNNGRU(self.num_classes).to(self.device)
            model.load_state_dict(torch.load("saved_models/cnn_gru.pth",
                                            map_location=self.device))
        model.eval()
        return model

    def count_parameters(self, model):
        """Count model parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def get_model_size(self, model_type):
        """Get model file size in MB"""
        path = f"saved_models/cnn_{model_type.lower()}.pth"
        if os.path.exists(path):
            size_bytes = os.path.getsize(path)
            return size_bytes / (1024 * 1024)
        return 0

    def measure_inference_speed(self, model, model_name, num_runs=100):
        """Measure inference latency and throughput"""
        print(f"\nMeasuring inference speed for {model_name} ({num_runs} runs)...")

        dummy_input = torch.randn(1, 15, 1, 112, 112).to(self.device)

        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)

        times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)  # ms

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000 / np.mean(times)
        }

    def compare_models(self):
        """
        Comprehensive comparison of BiLSTM and GRU models
        Includes all dissertation metrics
        """
        results = {}

        for model_type in ['bilstm', 'gru']:
            print(f"\n{'#'*70}")
            print(f"# ANALYZING {model_type.upper()} MODEL")
            print(f"{'#'*70}")

            model = self.load_model(model_type)

            total_params, trainable_params = self.count_parameters(model)
            model_size = self.get_model_size(model_type)

            print(f"\n--- Model Complexity ---")
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            print(f"Model Size: {model_size:.2f} MB")

            print(f"\n--- Calculating Dissertation Metrics ---")
            metrics = calculate_all_metrics(model, self.test_loader, self.device,
                                           self.idx2word, self.num_classes)

            print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
            print(f"Top-3 Accuracy: {metrics['top3_accuracy']:.2f}%")
            print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
            print(f"Word Error Rate (WER): {metrics['wer']:.2f}%")
            print(f"Character Error Rate (CER): {metrics['cer']:.2f}%")

            inference_stats = self.measure_inference_speed(model, model_type.upper())
            print(f"\n--- Inference Performance ---")
            print(f"Mean Latency: {inference_stats['mean_ms']:.2f} ms")
            print(f"Std Latency: {inference_stats['std_ms']:.2f} ms")
            print(f"FPS: {inference_stats['fps']:.2f}")

            results[model_type] = {
                'parameters': {
                    'total': total_params,
                    'trainable': trainable_params
                },
                'model_size_mb': model_size,
                'accuracy': {
                    'top1': metrics['top1_accuracy'],
                    'top3': metrics['top3_accuracy'],
                    'top5': metrics['top5_accuracy']
                },
                'wer': metrics['wer'],
                'cer': metrics['cer'],
                'wer_details': metrics['wer_details'],
                'cer_details': metrics['cer_details'],
                'per_class_metrics': {self.idx2word[k]: v for k, v in metrics['per_class_metrics'].items()},
                'confusion_matrix': metrics['confusion_matrix'],
                'inference': inference_stats,
                'predictions': [int(p) for p in metrics['predictions']],
                'labels': [int(l) for l in metrics['references']]
            }

        save_path = "saved_models/comparison_report.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved detailed comparison to {save_path}")

        self.print_comparison_summary(results)

        self.generate_comparison_plots(results)

        return results

    def print_comparison_summary(self, results):
        """Print side-by-side comparison"""
        print(f"\n{'='*70}")
        print(f"COMPARATIVE ANALYSIS SUMMARY (Dissertation)")
        print(f"{'='*70}\n")

        bilstm = results['bilstm']
        gru = results['gru']

        print(f"{'Metric':<30} {'BiLSTM':>15} {'GRU':>15} {'Difference':>15}")
        print("-" * 75)

        print("\n--- Accuracy Metrics ---")
        metrics_list = [
            ('Top-1 Accuracy (%)', bilstm['accuracy']['top1'], gru['accuracy']['top1']),
            ('Top-3 Accuracy (%)', bilstm['accuracy']['top3'], gru['accuracy']['top3']),
            ('Top-5 Accuracy (%)', bilstm['accuracy']['top5'], gru['accuracy']['top5']),
            ('Word Error Rate (%)', bilstm['wer'], gru['wer']),
            ('Character Error Rate (%)', bilstm['cer'], gru['cer']),
        ]

        for metric_name, bilstm_val, gru_val in metrics_list:
            diff = bilstm_val - gru_val
            print(f"{metric_name:<30} {bilstm_val:>15.2f} {gru_val:>15.2f} {diff:>+15.2f}")

        print("\n--- Model Complexity ---")
        print(f"{'Total Parameters':<30} {bilstm['parameters']['total']:>15,} {gru['parameters']['total']:>15,} {bilstm['parameters']['total'] - gru['parameters']['total']:>+15,}")
        print(f"{'Model Size (MB)':<30} {bilstm['model_size_mb']:>15.2f} {gru['model_size_mb']:>15.2f} {bilstm['model_size_mb'] - gru['model_size_mb']:>+15.2f}")

        print("\n--- Inference Performance ---")
        print(f"{'Mean Latency (ms)':<30} {bilstm['inference']['mean_ms']:>15.2f} {gru['inference']['mean_ms']:>15.2f} {bilstm['inference']['mean_ms'] - gru['inference']['mean_ms']:>+15.2f}")
        print(f"{'FPS':<30} {bilstm['inference']['fps']:>15.2f} {gru['inference']['fps']:>15.2f} {bilstm['inference']['fps'] - gru['inference']['fps']:>+15.2f}")

        print("\n--- Per-Class Top-1 Accuracy ---")
        print(f"{'Class':<15} {'BiLSTM':>15} {'GRU':>15} {'Difference':>15}")
        print("-" * 60)
        for word in sorted(bilstm['per_class_metrics'].keys()):
            bilstm_acc = bilstm['per_class_metrics'][word]['accuracy']
            gru_acc = gru['per_class_metrics'][word]['accuracy']
            diff = bilstm_acc - gru_acc
            print(f"{word:<15} {bilstm_acc:>15.2f} {gru_acc:>15.2f} {diff:>+15.2f}")

        print(f"\n{'='*70}")
        if bilstm['accuracy']['top1'] > gru['accuracy']['top1']:
            winner = "BiLSTM"
            margin = bilstm['accuracy']['top1'] - gru['accuracy']['top1']
        else:
            winner = "GRU"
            margin = gru['accuracy']['top1'] - bilstm['accuracy']['top1']

        print(f"🏆 Best Model (Top-1 Accuracy): {winner} (+{margin:.2f}%)")

        if bilstm['inference']['mean_ms'] < gru['inference']['mean_ms']:
            faster = "BiLSTM"
            speed_diff = gru['inference']['mean_ms'] - bilstm['inference']['mean_ms']
        else:
            faster = "GRU"
            speed_diff = bilstm['inference']['mean_ms'] - gru['inference']['mean_ms']

        print(f"⚡ Fastest Model (Latency): {faster} ({speed_diff:.2f} ms faster)")
        print(f"{'='*70}\n")

    def generate_comparison_plots(self, results):
        """Generate visualization plots for comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        bilstm = results['bilstm']
        gru = results['gru']

        ax1 = axes[0, 0]
        metrics = ['Top-1', 'Top-3', 'Top-5']
        bilstm_accs = [bilstm['accuracy']['top1'], bilstm['accuracy']['top3'], bilstm['accuracy']['top5']]
        gru_accs = [gru['accuracy']['top1'], gru['accuracy']['top3'], gru['accuracy']['top5']]

        x = np.arange(len(metrics))
        width = 0.35
        ax1.bar(x - width/2, bilstm_accs, width, label='BiLSTM', color='#3498db')
        ax1.bar(x + width/2, gru_accs, width, label='GRU', color='#e74c3c')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        error_metrics = ['WER', 'CER']
        bilstm_errors = [bilstm['wer'], bilstm['cer']]
        gru_errors = [gru['wer'], gru['cer']]

        x = np.arange(len(error_metrics))
        ax2.bar(x - width/2, bilstm_errors, width, label='BiLSTM', color='#3498db')
        ax2.bar(x + width/2, gru_errors, width, label='GRU', color='#e74c3c')
        ax2.set_ylabel('Error Rate (%)')
        ax2.set_title('Error Rate Comparison (Lower is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(error_metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        complexity_metrics = ['Parameters\n(Millions)', 'Model Size\n(MB)']
        bilstm_complexity = [bilstm['parameters']['total'] / 1e6, bilstm['model_size_mb']]
        gru_complexity = [gru['parameters']['total'] / 1e6, gru['model_size_mb']]

        x = np.arange(len(complexity_metrics))
        ax3.bar(x - width/2, bilstm_complexity, width, label='BiLSTM', color='#3498db')
        ax3.bar(x + width/2, gru_complexity, width, label='GRU', color='#e74c3c')
        ax3.set_ylabel('Value')
        ax3.set_title('Model Complexity')
        ax3.set_xticks(x)
        ax3.set_xticklabels(complexity_metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        speed_metrics = ['Latency (ms)', 'FPS']
        bilstm_speed = [bilstm['inference']['mean_ms'], bilstm['inference']['fps']]
        gru_speed = [gru['inference']['mean_ms'], gru['inference']['fps']]

        ax4_twin = ax4.twinx()
        x = np.arange(2)
        bars1 = ax4.bar([0 - width/2], [bilstm_speed[0]], width, label='BiLSTM', color='#3498db')
        bars2 = ax4.bar([0 + width/2], [gru_speed[0]], width, label='GRU', color='#e74c3c')
        ax4.set_ylabel('Latency (ms)')
        ax4.set_title('Inference Performance')
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Latency', 'FPS'])
        ax4.grid(True, alpha=0.3, axis='y')

        bars3 = ax4_twin.bar([1 - width/2], [bilstm_speed[1]], width, color='#3498db')
        bars4 = ax4_twin.bar([1 + width/2], [gru_speed[1]], width, color='#e74c3c')
        ax4_twin.set_ylabel('FPS')

        ax4.legend(loc='upper left')

        plt.suptitle('BiLSTM vs GRU: Comprehensive Comparison (Dissertation)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = 'saved_models/model_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
        plt.close()


if __name__ == "__main__":
    analyzer = ModelAnalyzer()
    results = analyzer.compare_models()
    print("\n✅ Analysis Complete!")
