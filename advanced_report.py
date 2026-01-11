"""
Advanced Training Report and Model Comparison
Comprehensive analysis of BiLSTM vs GRU with detailed metrics, visualizations,
and recommendations for improvement.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from src.models.cnn_bilstm import CNNBiLSTM
from src.models.cnn_gru import CNNGRU


class AdvancedModelReport:
    """Generate comprehensive training analysis and comparison report"""
    
    def __init__(self):
        self.bilstm_history = None
        self.gru_history = None
        self.labels = None
        
        # Load training histories
        self._load_histories()
        
    def _load_histories(self):
        """Load training history files"""
        bilstm_path = "saved_models/bilstm_training_history.json"
        gru_path = "saved_models/gru_training_history.json"
        labels_path = "saved_models/labels.json"
        
        if os.path.exists(bilstm_path):
            with open(bilstm_path, 'r') as f:
                self.bilstm_history = json.load(f)
        else:
            print(f"⚠ Warning: {bilstm_path} not found")
            
        if os.path.exists(gru_path):
            with open(gru_path, 'r') as f:
                self.gru_history = json.load(f)
        else:
            print(f"⚠ Warning: {gru_path} not found")
            
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
    
    def analyze_overfitting(self, history, model_name):
        """Detect and quantify overfitting"""
        if not history:
            return None
            
        train_acc = np.array(history['train_acc'])
        val_acc = np.array(history['val_acc'])
        train_loss = np.array(history['train_loss'])
        val_loss = np.array(history['val_loss'])
        
        # Calculate metrics
        final_gap = train_acc[-1] - val_acc[-1]
        max_val_acc = val_acc.max()
        max_val_epoch = val_acc.argmax() + 1
        
        # Detect when validation loss starts increasing
        val_loss_trend = val_loss[len(val_loss)//2:] - val_loss[len(val_loss)//2]
        increasing_loss = val_loss_trend > 0
        
        # Early stopping recommendation
        early_stop_epoch = max_val_epoch
        for i in range(max_val_epoch, len(val_acc)):
            if val_acc[i] < max_val_acc - 1.0:  # 1% drop tolerance
                early_stop_epoch = i
                break
        
        analysis = {
            'model': model_name,
            'total_epochs': len(train_acc),
            'final_train_acc': train_acc[-1],
            'final_val_acc': val_acc[-1],
            'overfitting_gap': final_gap,
            'max_val_acc': max_val_acc,
            'max_val_epoch': max_val_epoch,
            'early_stop_recommendation': early_stop_epoch,
            'epochs_wasted': len(train_acc) - early_stop_epoch,
            'val_loss_increasing': bool(increasing_loss.sum() > len(increasing_loss) * 0.6),
            'overfitting_severity': self._classify_overfitting(final_gap)
        }
        
        return analysis
    
    def _classify_overfitting(self, gap):
        """Classify overfitting severity"""
        if gap < 5:
            return "None/Minimal"
        elif gap < 10:
            return "Mild"
        elif gap < 15:
            return "Moderate"
        elif gap < 20:
            return "Severe"
        else:
            return "Critical"
    
    def generate_training_curves(self):
        """Generate comprehensive training curves visualization"""
        if not self.bilstm_history and not self.gru_history:
            print("No training history available")
            return
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metric_comparison(ax1, 'train_acc', 'Training Accuracy', 'Accuracy (%)')
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_metric_comparison(ax2, 'val_acc', 'Validation Accuracy', 'Accuracy (%)')
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_overfitting_gap(ax3)
        
        # Row 2: Loss comparison
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_metric_comparison(ax4, 'train_loss', 'Training Loss', 'Loss')
        
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_metric_comparison(ax5, 'val_loss', 'Validation Loss', 'Loss')
        
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_loss_landscape(ax6)
        
        # Row 3: Advanced metrics
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_generalization_gap(ax7)
        
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_learning_rate_effect(ax8)
        
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_convergence_speed(ax9)
        
        plt.suptitle('Comprehensive Training Analysis: BiLSTM vs GRU', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('saved_models/advanced_training_report.png', dpi=300, bbox_inches='tight')
        print("Saved visualization to saved_models/advanced_training_report.png")
        plt.close()
    
    def _plot_metric_comparison(self, ax, metric, title, ylabel):
        """Plot metric comparison between models"""
        if self.bilstm_history and metric in self.bilstm_history:
            epochs = range(1, len(self.bilstm_history[metric]) + 1)
            ax.plot(epochs, self.bilstm_history[metric], 
                   label='BiLSTM', linewidth=2, marker='o', markersize=3)
        
        if self.gru_history and metric in self.gru_history:
            epochs = range(1, len(self.gru_history[metric]) + 1)
            ax.plot(epochs, self.gru_history[metric], 
                   label='GRU', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_overfitting_gap(self, ax):
        """Plot train-val accuracy gap over time"""
        if self.bilstm_history:
            bilstm_gap = np.array(self.bilstm_history['train_acc']) - np.array(self.bilstm_history['val_acc'])
            epochs = range(1, len(bilstm_gap) + 1)
            ax.plot(epochs, bilstm_gap, label='BiLSTM Gap', linewidth=2, marker='o', markersize=3)
        
        if self.gru_history:
            gru_gap = np.array(self.gru_history['train_acc']) - np.array(self.gru_history['val_acc'])
            epochs = range(1, len(gru_gap) + 1)
            ax.plot(epochs, gru_gap, label='GRU Gap', linewidth=2, marker='s', markersize=3)
        
        # Add warning zones
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Mild Overfitting')
        ax.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Severe Overfitting')
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Train-Val Gap (%)', fontsize=10)
        ax.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_loss_landscape(self, ax):
        """Plot train vs val loss relationship"""
        if self.bilstm_history:
            ax.scatter(self.bilstm_history['train_loss'], 
                      self.bilstm_history['val_loss'],
                      c=range(len(self.bilstm_history['train_loss'])),
                      cmap='viridis', label='BiLSTM', marker='o', s=50, alpha=0.6)
        
        if self.gru_history:
            ax.scatter(self.gru_history['train_loss'], 
                      self.gru_history['val_loss'],
                      c=range(len(self.gru_history['train_loss'])),
                      cmap='plasma', label='GRU', marker='s', s=50, alpha=0.6)
        
        # Add ideal line
        max_loss = max(
            max(self.bilstm_history.get('train_loss', [0])),
            max(self.gru_history.get('train_loss', [0]))
        )
        ax.plot([0, max_loss], [0, max_loss], 'k--', alpha=0.3, label='Ideal (no overfitting)')
        
        ax.set_xlabel('Training Loss', fontsize=10)
        ax.set_ylabel('Validation Loss', fontsize=10)
        ax.set_title('Loss Landscape (color = epoch)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_generalization_gap(self, ax):
        """Plot generalization gap trend"""
        if self.bilstm_history:
            train_acc = np.array(self.bilstm_history['train_acc'])
            val_acc = np.array(self.bilstm_history['val_acc'])
            generalization = 100 * (1 - val_acc / train_acc)
            epochs = range(1, len(generalization) + 1)
            ax.plot(epochs, generalization, label='BiLSTM', linewidth=2, marker='o', markersize=3)
        
        if self.gru_history:
            train_acc = np.array(self.gru_history['train_acc'])
            val_acc = np.array(self.gru_history['val_acc'])
            generalization = 100 * (1 - val_acc / train_acc)
            epochs = range(1, len(generalization) + 1)
            ax.plot(epochs, generalization, label='GRU', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Generalization Gap (%)', fontsize=10)
        ax.set_title('Generalization Performance', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_rate_effect(self, ax):
        """Plot learning progress rate"""
        if self.bilstm_history:
            val_acc = np.array(self.bilstm_history['val_acc'])
            improvement = np.diff(val_acc, prepend=val_acc[0])
            epochs = range(1, len(improvement) + 1)
            ax.plot(epochs, improvement, label='BiLSTM Δ Acc', linewidth=2, marker='o', markersize=3)
        
        if self.gru_history:
            val_acc = np.array(self.gru_history['val_acc'])
            improvement = np.diff(val_acc, prepend=val_acc[0])
            epochs = range(1, len(improvement) + 1)
            ax.plot(epochs, improvement, label='GRU Δ Acc', linewidth=2, marker='s', markersize=3)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Accuracy Improvement (%)', fontsize=10)
        ax.set_title('Learning Rate (per epoch)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_speed(self, ax):
        """Plot convergence speed comparison"""
        targets = [50, 60, 70, 80]
        
        bilstm_epochs = []
        gru_epochs = []
        
        if self.bilstm_history:
            val_acc = self.bilstm_history['val_acc']
            for target in targets:
                epoch = next((i+1 for i, acc in enumerate(val_acc) if acc >= target), len(val_acc))
                bilstm_epochs.append(epoch)
        
        if self.gru_history:
            val_acc = self.gru_history['val_acc']
            for target in targets:
                epoch = next((i+1 for i, acc in enumerate(val_acc) if acc >= target), len(val_acc))
                gru_epochs.append(epoch)
        
        x = np.arange(len(targets))
        width = 0.35
        
        if bilstm_epochs:
            ax.bar(x - width/2, bilstm_epochs, width, label='BiLSTM', alpha=0.8)
        if gru_epochs:
            ax.bar(x + width/2, gru_epochs, width, label='GRU', alpha=0.8)
        
        ax.set_xlabel('Target Accuracy (%)', fontsize=10)
        ax.set_ylabel('Epochs to Reach', fontsize=10)
        ax.set_title('Convergence Speed', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t}%' for t in targets])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def print_comprehensive_report(self):
        """Print detailed text report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TRAINING ANALYSIS REPORT".center(80))
        print("="*80 + "\n")
        
        # Analyze both models
        if self.bilstm_history:
            bilstm_analysis = self.analyze_overfitting(self.bilstm_history, "BiLSTM")
            self._print_model_analysis(bilstm_analysis)
        
        if self.gru_history:
            gru_analysis = self.analyze_overfitting(self.gru_history, "GRU")
            self._print_model_analysis(gru_analysis)
        
        # Comparison
        if self.bilstm_history and self.gru_history:
            self._print_comparison(bilstm_analysis, gru_analysis)
        
        # Recommendations
        self._print_recommendations(bilstm_analysis if self.bilstm_history else gru_analysis)
        
        print("\n" + "="*80)
    
    def _print_model_analysis(self, analysis):
        """Print analysis for single model"""
        if not analysis:
            return
            
        print(f"\n{'─'*80}")
        print(f"{analysis['model']} MODEL ANALYSIS".center(80))
        print(f"{'─'*80}\n")
        
        print(f"Training Configuration:")
        print(f"  Total Epochs Trained: {analysis['total_epochs']}")
        print(f"  Optimal Stop Point: Epoch {analysis['max_val_epoch']} (saved {analysis['epochs_wasted']} epochs)")
        
        print(f"\nPerformance Metrics:")
        print(f"  Final Training Accuracy: {analysis['final_train_acc']:.2f}%")
        print(f"  Final Validation Accuracy: {analysis['final_val_acc']:.2f}%")
        print(f"  Best Validation Accuracy: {analysis['max_val_acc']:.2f}% (Epoch {analysis['max_val_epoch']})")
        
        print(f"\nOverfitting Analysis:")
        print(f"  Train-Val Gap: {analysis['overfitting_gap']:.2f}%")
        print(f"  Severity: {analysis['overfitting_severity']}")
        print(f"  Validation Loss Trend: {'Increasing ⚠️' if analysis['val_loss_increasing'] else 'Stable ✓'}")
        
        # Status indicator
        if analysis['overfitting_severity'] in ['Critical', 'Severe']:
            status = "🔴 CRITICAL - Significant overfitting detected"
        elif analysis['overfitting_severity'] == 'Moderate':
            status = "🟡 WARNING - Moderate overfitting"
        else:
            status = "🟢 GOOD - Minimal overfitting"
        
        print(f"\n  Status: {status}")
        
        if analysis['epochs_wasted'] > 5:
            print(f"\n  ⚠️  Recommendation: Use early stopping to save {analysis['epochs_wasted']} epochs")
    
    def _print_comparison(self, bilstm, gru):
        """Print head-to-head comparison"""
        print(f"\n{'─'*80}")
        print("HEAD-TO-HEAD COMPARISON".center(80))
        print(f"{'─'*80}\n")
        
        print(f"{'Metric':<40} {'BiLSTM':>15} {'GRU':>15} {'Winner':>10}")
        print(f"{'-'*80}")
        
        # Best validation accuracy
        winner = 'BiLSTM' if bilstm['max_val_acc'] > gru['max_val_acc'] else 'GRU'
        print(f"{'Best Validation Accuracy':<40} {bilstm['max_val_acc']:>14.2f}% {gru['max_val_acc']:>14.2f}% {winner:>10}")
        
        # Overfitting resistance (lower gap is better)
        winner = 'BiLSTM' if bilstm['overfitting_gap'] < gru['overfitting_gap'] else 'GRU'
        print(f"{'Overfitting Resistance (lower better)':<40} {bilstm['overfitting_gap']:>14.2f}% {gru['overfitting_gap']:>14.2f}% {winner:>10}")
        
        # Convergence speed (fewer epochs to peak)
        winner = 'BiLSTM' if bilstm['max_val_epoch'] < gru['max_val_epoch'] else 'GRU'
        print(f"{'Convergence Speed (epochs to peak)':<40} {bilstm['max_val_epoch']:>15d} {gru['max_val_epoch']:>15d} {winner:>10}")
        
        # Training efficiency (less wasted epochs)
        winner = 'BiLSTM' if bilstm['epochs_wasted'] < gru['epochs_wasted'] else 'GRU'
        print(f"{'Training Efficiency (wasted epochs)':<40} {bilstm['epochs_wasted']:>15d} {gru['epochs_wasted']:>15d} {winner:>10}")
        
        print(f"{'-'*80}")
        
        # Overall assessment
        bilstm_score = sum([
            bilstm['max_val_acc'] > gru['max_val_acc'],
            bilstm['overfitting_gap'] < gru['overfitting_gap'],
            bilstm['max_val_epoch'] < gru['max_val_epoch'],
            bilstm['epochs_wasted'] < gru['epochs_wasted']
        ])
        
        if bilstm_score > 2:
            print(f"\n🏆 Overall Winner: BiLSTM ({bilstm_score}/4 metrics)")
        elif bilstm_score < 2:
            print(f"\n🏆 Overall Winner: GRU ({4-bilstm_score}/4 metrics)")
        else:
            print(f"\n🤝 Tie: Both models perform comparably")
    
    def _print_recommendations(self, analysis):
        """Print improvement recommendations"""
        print(f"\n{'─'*80}")
        print("RECOMMENDATIONS FOR IMPROVEMENT".center(80))
        print(f"{'─'*80}\n")
        
        recommendations = []
        
        # Based on overfitting
        if analysis['overfitting_gap'] > 10:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Severe overfitting detected',
                'solution': 'Add dropout (0.3-0.5), increase weight decay, use data augmentation'
            })
        
        # Based on wasted epochs
        if analysis['epochs_wasted'] > 5:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f'Training continued {analysis["epochs_wasted"]} epochs past optimal point',
                'solution': f'Implement early stopping with patience=5, monitor validation loss'
            })
        
        # Based on validation loss trend
        if analysis['val_loss_increasing']:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': 'Validation loss increasing in later epochs',
                'solution': 'Reduce learning rate, add L2 regularization (weight_decay=1e-4)'
            })
        
        # General improvements
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': 'Enhance model generalization',
            'solution': 'Add data augmentation: random crop, rotation, brightness/contrast'
        })
        
        recommendations.append({
            'priority': 'LOW',
            'issue': 'Optimize training time',
            'solution': 'Use mixed precision training (torch.cuda.amp), increase batch size if GPU memory allows'
        })
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            priority_color = {
                'HIGH': '🔴',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }
            
            print(f"{i}. {priority_color[rec['priority']]} [{rec['priority']}] {rec['issue']}")
            print(f"   Solution: {rec['solution']}\n")
        
        
    
    def save_detailed_report(self):
        """Save detailed report to file"""
        report_path = "saved_models/detailed_training_analysis.txt"
        
        # Redirect print to file
        import sys
        original_stdout = sys.stdout
        
        with open(report_path, 'w') as f:
            sys.stdout = f
            self.print_comprehensive_report()
            sys.stdout = original_stdout
        
        print(f"\nDetailed report saved to {report_path}")


def main():
    print("\n" + "="*80)
    print("ADVANCED TRAINING REPORT GENERATOR".center(80))
    print("="*80)
    
    report = AdvancedModelReport()
    
    # Generate visualizations
    print("\nGenerating advanced visualizations...")
    report.generate_training_curves()
    
    # Print comprehensive report
    report.print_comprehensive_report()
    
    # Save to file
    report.save_detailed_report()
    
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE".center(80))
    print("="*80)
    print("\nGenerated files:")
    print("   saved_models/advanced_training_report.png")
    print("  📄 saved_models/detailed_training_analysis.txt")
    print("\n")


if __name__ == "__main__":
    main()
