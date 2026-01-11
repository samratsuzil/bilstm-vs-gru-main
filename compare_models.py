#!/usr/bin/env python3
"""
Comparative Analysis Script for Bi-LSTM vs GRU Lip Reading Models

This script generates a comprehensive comparison report aligned with the
research objectives from the proposal:
- Recognition Accuracy
- Training Time
- Inference Latency
- Model Complexity

Author: Sushil Subedi
Institution: Tribhuvan University, Institute of Science and Technology
"""

import json
import os
from datetime import datetime

def load_training_history(model_type):
    """Load training history JSON for a given model"""
    path = f"saved_models/{model_type}_training_history.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def format_number(num):
    """Format large numbers with commas"""
    return f"{int(num):,}"

def compare_models():
    """Generate comprehensive comparison between Bi-LSTM and GRU models"""

    print("\n" + "="*80)
    print("LOADING MODEL TRAINING HISTORIES")
    print("="*80)

    bilstm_history = load_training_history('bilstm')
    gru_history = load_training_history('gru')

    if bilstm_history is None:
        print("ERROR: Bi-LSTM training history not found!")
        print("Please train the Bi-LSTM model first using:")
        print("  python train.py --model bilstm")
        return

    if gru_history is None:
        print("ERROR: GRU training history not found!")
        print("Please train the GRU model first using:")
        print("  python train.py --model gru")
        return

    print("✓ Bi-LSTM history loaded")
    print("✓ GRU history loaded")
    print()

    # Generate comparison report
    report_lines = []

    report_lines.append("="*80)
    report_lines.append("COMPARATIVE STUDY OF BI-LSTM AND GRU ARCHITECTURES FOR LIP READING")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("RESEARCH CONTEXT")
    report_lines.append("-"*80)
    report_lines.append("This report presents a systematic comparison of Bidirectional LSTM and")
    report_lines.append("Gated Recurrent Unit architectures for visual-only speech recognition")
    report_lines.append("(lip reading). Both models use identical CNN backbones and are evaluated")
    report_lines.append("under the same experimental conditions.")
    report_lines.append("")
    report_lines.append("Dataset: Lip Reading in the Wild (LRW)")
    report_lines.append("Institution: Tribhuvan University")
    report_lines.append("Author: Sushil Subedi (Roll No: 26/076)")
    report_lines.append("")

    # Model Architecture Comparison
    report_lines.append("="*80)
    report_lines.append("1. MODEL ARCHITECTURE COMPARISON")
    report_lines.append("="*80)
    report_lines.append("")

    bilstm_params = bilstm_history.get('total_parameters', 0)
    gru_params = gru_history.get('total_parameters', 0)
    bilstm_size = bilstm_history.get('model_size_mb', 0)
    gru_size = gru_history.get('model_size_mb', 0)

    report_lines.append(f"{'Metric':<30} {'Bi-LSTM':<20} {'GRU':<20} {'Winner':<10}")
    report_lines.append("-"*80)
    report_lines.append(f"{'Total Parameters':<30} {format_number(bilstm_params):<20} {format_number(gru_params):<20} {'GRU' if gru_params < bilstm_params else 'Bi-LSTM':<10}")
    report_lines.append(f"{'Model Size (MB)':<30} {bilstm_size:<20.2f} {gru_size:<20.2f} {'GRU' if gru_size < bilstm_size else 'Bi-LSTM':<10}")
    report_lines.append("")

    param_reduction = ((bilstm_params - gru_params) / bilstm_params * 100) if bilstm_params > 0 else 0
    size_reduction = ((bilstm_size - gru_size) / bilstm_size * 100) if bilstm_size > 0 else 0

    report_lines.append(f"Parameter Reduction (GRU vs Bi-LSTM): {param_reduction:.2f}%")
    report_lines.append(f"Size Reduction (GRU vs Bi-LSTM): {size_reduction:.2f}%")
    report_lines.append("")

    # Recognition Accuracy Comparison
    report_lines.append("="*80)
    report_lines.append("2. RECOGNITION ACCURACY COMPARISON (Research Target: 85-95%)")
    report_lines.append("="*80)
    report_lines.append("")

    bilstm_acc = bilstm_history.get('test_acc', 0)
    gru_acc = gru_history.get('test_acc', 0)
    bilstm_top3 = bilstm_history.get('test_top3_acc', 0)
    gru_top3 = gru_history.get('test_top3_acc', 0)
    bilstm_top5 = bilstm_history.get('test_top5_acc', 0)
    gru_top5 = gru_history.get('test_top5_acc', 0)
    bilstm_wer = bilstm_history.get('test_wer', 100)
    gru_wer = gru_history.get('test_wer', 100)
    bilstm_cer = bilstm_history.get('test_cer', 100)
    gru_cer = gru_history.get('test_cer', 100)

    report_lines.append(f"{'Metric':<30} {'Bi-LSTM (%)':<20} {'GRU (%)':<20} {'Winner':<10}")
    report_lines.append("-"*80)
    report_lines.append(f"{'Top-1 Accuracy':<30} {bilstm_acc:<20.2f} {gru_acc:<20.2f} {'Bi-LSTM' if bilstm_acc > gru_acc else 'GRU':<10}")
    report_lines.append(f"{'Top-3 Accuracy':<30} {bilstm_top3:<20.2f} {gru_top3:<20.2f} {'Bi-LSTM' if bilstm_top3 > gru_top3 else 'GRU':<10}")
    report_lines.append(f"{'Top-5 Accuracy':<30} {bilstm_top5:<20.2f} {gru_top5:<20.2f} {'Bi-LSTM' if bilstm_top5 > gru_top5 else 'GRU':<10}")
    report_lines.append(f"{'Word Error Rate (WER)':<30} {bilstm_wer:<20.2f} {gru_wer:<20.2f} {'Bi-LSTM' if bilstm_wer < gru_wer else 'GRU':<10}")
    report_lines.append(f"{'Character Error Rate (CER)':<30} {bilstm_cer:<20.2f} {gru_cer:<20.2f} {'Bi-LSTM' if bilstm_cer < gru_cer else 'GRU':<10}")
    report_lines.append("")

    acc_diff = bilstm_acc - gru_acc
    report_lines.append(f"Accuracy Difference: {abs(acc_diff):.2f}% ({'Bi-LSTM leads' if acc_diff > 0 else 'GRU leads' if acc_diff < 0 else 'Tied'})")

    bilstm_meets_target = bilstm_acc >= 85.0 and bilstm_acc <= 95.0
    gru_meets_target = gru_acc >= 85.0 and gru_acc <= 95.0

    report_lines.append("")
    report_lines.append("Research Objective Status (85-95% Target):")
    report_lines.append(f"  Bi-LSTM: {'✓ ACHIEVED' if bilstm_meets_target else '✗ NOT MET'} ({bilstm_acc:.2f}%)")
    report_lines.append(f"  GRU:     {'✓ ACHIEVED' if gru_meets_target else '✗ NOT MET'} ({gru_acc:.2f}%)")
    report_lines.append("")

    # Training Efficiency Comparison
    report_lines.append("="*80)
    report_lines.append("3. TRAINING EFFICIENCY COMPARISON")
    report_lines.append("="*80)
    report_lines.append("")

    bilstm_train_time = bilstm_history.get('total_training_time_minutes', 0)
    gru_train_time = gru_history.get('total_training_time_minutes', 0)

    report_lines.append(f"{'Metric':<30} {'Bi-LSTM':<20} {'GRU':<20} {'Winner':<10}")
    report_lines.append("-"*80)
    report_lines.append(f"{'Training Time (minutes)':<30} {bilstm_train_time:<20.1f} {gru_train_time:<20.1f} {'GRU' if gru_train_time < bilstm_train_time else 'Bi-LSTM':<10}")

    time_savings = ((bilstm_train_time - gru_train_time) / bilstm_train_time * 100) if bilstm_train_time > 0 else 0
    report_lines.append("")
    report_lines.append(f"Training Time Reduction (GRU vs Bi-LSTM): {time_savings:.2f}%")
    report_lines.append("")

    # Inference Performance Comparison
    report_lines.append("="*80)
    report_lines.append("4. REAL-TIME INFERENCE PERFORMANCE (Target: <200ms latency, ≥30 FPS)")
    report_lines.append("="*80)
    report_lines.append("")

    bilstm_inf_time = bilstm_history.get('avg_inference_time_ms', 0)
    gru_inf_time = gru_history.get('avg_inference_time_ms', 0)
    bilstm_fps = bilstm_history.get('fps', 0)
    gru_fps = gru_history.get('fps', 0)

    report_lines.append(f"{'Metric':<30} {'Bi-LSTM':<20} {'GRU':<20} {'Winner':<10}")
    report_lines.append("-"*80)
    report_lines.append(f"{'Avg Inference Time (ms)':<30} {bilstm_inf_time:<20.2f} {gru_inf_time:<20.2f} {'GRU' if gru_inf_time < bilstm_inf_time else 'Bi-LSTM':<10}")
    report_lines.append(f"{'Frames Per Second (FPS)':<30} {bilstm_fps:<20.2f} {gru_fps:<20.2f} {'GRU' if gru_fps > bilstm_fps else 'Bi-LSTM':<10}")
    report_lines.append("")

    speedup = (bilstm_inf_time / gru_inf_time) if gru_inf_time > 0 else 0
    report_lines.append(f"Inference Speedup (GRU vs Bi-LSTM): {speedup:.2f}x")

    bilstm_realtime_latency = bilstm_inf_time < 200
    gru_realtime_latency = gru_inf_time < 200
    bilstm_realtime_fps = bilstm_fps >= 30
    gru_realtime_fps = gru_fps >= 30

    report_lines.append("")
    report_lines.append("Real-Time Capability Assessment:")
    report_lines.append(f"  Bi-LSTM:")
    report_lines.append(f"    Latency Target (<200ms): {'✓ ACHIEVED' if bilstm_realtime_latency else '✗ NOT MET'} ({bilstm_inf_time:.2f}ms)")
    report_lines.append(f"    FPS Target (≥30):        {'✓ ACHIEVED' if bilstm_realtime_fps else '✗ NOT MET'} ({bilstm_fps:.2f} FPS)")
    report_lines.append(f"  GRU:")
    report_lines.append(f"    Latency Target (<200ms): {'✓ ACHIEVED' if gru_realtime_latency else '✗ NOT MET'} ({gru_inf_time:.2f}ms)")
    report_lines.append(f"    FPS Target (≥30):        {'✓ ACHIEVED' if gru_realtime_fps else '✗ NOT MET'} ({gru_fps:.2f} FPS)")
    report_lines.append("")

    # Overall Trade-off Analysis
    report_lines.append("="*80)
    report_lines.append("5. OVERALL TRADE-OFF ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")

    # Calculate scores
    bilstm_score = 0
    gru_score = 0

    # Accuracy (weight: 40%)
    if bilstm_acc > gru_acc:
        bilstm_score += 40
    elif gru_acc > bilstm_acc:
        gru_score += 40
    else:
        bilstm_score += 20
        gru_score += 20

    # Efficiency (weight: 30%)
    if gru_params < bilstm_params:
        gru_score += 30
    else:
        bilstm_score += 30

    # Inference Speed (weight: 30%)
    if gru_fps > bilstm_fps:
        gru_score += 30
    else:
        bilstm_score += 30

    report_lines.append("Weighted Score (Accuracy: 40%, Efficiency: 30%, Speed: 30%):")
    report_lines.append(f"  Bi-LSTM: {bilstm_score}/100")
    report_lines.append(f"  GRU:     {gru_score}/100")
    report_lines.append("")

    if bilstm_score > gru_score:
        winner = "Bi-LSTM"
        explanation = "Bi-LSTM demonstrates superior performance, likely due to bidirectional context modeling."
    elif gru_score > bilstm_score:
        winner = "GRU"
        explanation = "GRU achieves better overall performance with improved efficiency and speed."
    else:
        winner = "Tie"
        explanation = "Both models show comparable performance with different trade-offs."

    report_lines.append(f"Overall Winner: {winner}")
    report_lines.append(f"Explanation: {explanation}")
    report_lines.append("")

    # Recommendations
    report_lines.append("="*80)
    report_lines.append("6. RECOMMENDATIONS")
    report_lines.append("="*80)
    report_lines.append("")

    if bilstm_acc - gru_acc > 2.0:
        report_lines.append("• Bi-LSTM shows significantly better accuracy (+{:.2f}%).".format(bilstm_acc - gru_acc))
        report_lines.append("  Recommended for: Applications where accuracy is critical.")
    elif gru_acc - bilstm_acc > 2.0:
        report_lines.append("• GRU shows significantly better accuracy (+{:.2f}%).".format(gru_acc - bilstm_acc))
        report_lines.append("  Recommended for: Applications where accuracy is critical.")
    else:
        report_lines.append("• Accuracy difference is marginal ({:.2f}%).".format(abs(bilstm_acc - gru_acc)))

    report_lines.append("")

    if gru_fps > bilstm_fps * 1.2:
        report_lines.append("• GRU provides significantly better inference speed ({:.2f}x faster).".format(gru_fps / bilstm_fps if bilstm_fps > 0 else 0))
        report_lines.append("  Recommended for: Real-time applications, edge devices.")
    elif bilstm_fps > gru_fps * 1.2:
        report_lines.append("• Bi-LSTM provides significantly better inference speed ({:.2f}x faster).".format(bilstm_fps / gru_fps if gru_fps > 0 else 0))
        report_lines.append("  Recommended for: Real-time applications, edge devices.")
    else:
        report_lines.append("• Inference speeds are comparable.")

    report_lines.append("")

    if gru_params < bilstm_params * 0.8:
        report_lines.append("• GRU has significantly fewer parameters ({:.2f}% reduction).".format(param_reduction))
        report_lines.append("  Recommended for: Resource-constrained environments, mobile deployment.")

    report_lines.append("")

    # Research Contribution
    report_lines.append("="*80)
    report_lines.append("7. RESEARCH CONTRIBUTION")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("This comparative study provides empirical evidence for:")
    report_lines.append("")
    report_lines.append("1. Performance Trade-offs: Quantifies the accuracy vs efficiency trade-off")
    report_lines.append("   between Bi-LSTM and GRU architectures for lip reading.")
    report_lines.append("")
    report_lines.append("2. Real-Time Feasibility: Demonstrates which architecture is more suitable")
    report_lines.append("   for real-time visual speech recognition applications.")
    report_lines.append("")
    report_lines.append("3. Deployment Guidance: Offers practical recommendations for selecting")
    report_lines.append("   the appropriate model based on application requirements.")
    report_lines.append("")
    report_lines.append("4. Reproducible Methodology: Provides a systematic evaluation framework")
    report_lines.append("   under controlled experimental conditions.")
    report_lines.append("")

    # Conclusion
    report_lines.append("="*80)
    report_lines.append("8. CONCLUSION")
    report_lines.append("="*80)
    report_lines.append("")

    both_meet_accuracy = bilstm_meets_target and gru_meets_target
    both_realtime = (bilstm_realtime_latency and bilstm_realtime_fps) and (gru_realtime_latency and gru_realtime_fps)

    if both_meet_accuracy and both_realtime:
        report_lines.append("Both models successfully meet the research objectives:")
        report_lines.append(f"• Accuracy target (85-95%): Bi-LSTM {bilstm_acc:.2f}%, GRU {gru_acc:.2f}%")
        report_lines.append(f"• Real-time capability: Both models achieve <200ms latency and ≥30 FPS")
        report_lines.append("")
        report_lines.append(f"The choice between Bi-LSTM and GRU depends on specific deployment")
        report_lines.append(f"requirements. Use Bi-LSTM for maximum accuracy, and GRU for")
        report_lines.append(f"better efficiency and faster inference.")
    else:
        report_lines.append("Research Objectives Status:")
        if bilstm_meets_target:
            report_lines.append(f"• Bi-LSTM meets accuracy target: {bilstm_acc:.2f}%")
        else:
            report_lines.append(f"• Bi-LSTM below accuracy target: {bilstm_acc:.2f}% (needs improvement)")

        if gru_meets_target:
            report_lines.append(f"• GRU meets accuracy target: {gru_acc:.2f}%")
        else:
            report_lines.append(f"• GRU below accuracy target: {gru_acc:.2f}% (needs improvement)")

        report_lines.append("")
        report_lines.append("Recommendations for improvement:")
        report_lines.append("• Increase training epochs (current architecture supports 80-100 epochs)")
        report_lines.append("• Use ResNet50 backbone for better feature extraction")
        report_lines.append("• Ensure aggressive data augmentation is enabled")
        report_lines.append("• Verify dataset quality and completeness")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF COMPARATIVE ANALYSIS REPORT")
    report_lines.append("="*80)

    # Write to file
    report_content = "\n".join(report_lines)
    output_path = "saved_models/comparative_analysis.txt"

    with open(output_path, 'w') as f:
        f.write(report_content)

    # Print to console
    print(report_content)

    print(f"\n✓ Comparative analysis saved to: {output_path}\n")

if __name__ == "__main__":
    compare_models()
