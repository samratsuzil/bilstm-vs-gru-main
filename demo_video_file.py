"""
Video File Lip Reading Demo
Process a pre-recorded video file and detect the spoken word.

Perfect for thesis presentation - use test videos from your dataset
where the model performs at 88.7% accuracy!

Usage:
    python3 demo_video_file.py --video path/to/video.mp4
    python3 demo_video_file.py --video path/to/video.mp4 --model gru
"""

import cv2
import torch
import json
import numpy as np
import argparse
import os
from src.models.cnn_bilstm import CNNBiLSTM
from src.models.cnn_gru import CNNGRU

# Configuration
SEQ_LEN = 15
IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_type='bilstm'):
    """Load trained model"""
    with open("saved_models/labels.json", "r") as f:
        word2idx = json.load(f)
    
    idx2word = {v: k for k, v in word2idx.items()}
    num_classes = len(word2idx)
    
    if model_type.lower() == 'bilstm':
        model = CNNBiLSTM(num_classes).to(DEVICE)
        model.load_state_dict(torch.load("saved_models/cnn_bilstm.pth", map_location=DEVICE))
        model_name = "BiLSTM"
    else:
        model = CNNGRU(num_classes).to(DEVICE)
        model.load_state_dict(torch.load("saved_models/cnn_gru.pth", map_location=DEVICE))
        model_name = "GRU"
    
    model.eval()
    return model, idx2word, model_name


def process_video(video_path, num_frames=SEQ_LEN):
    """
    Process video file and extract frames
    Same preprocessing as training/testing pipeline
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    print(f"Reading video: {os.path.basename(video_path)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        
        normalized = resized.astype(np.float32) / 255.0
        
        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
        frames.append(tensor)
    
    cap.release()
    
    if len(frames) == 0:
        print(" Error: No frames extracted from video")
        return None
    
    print(f"Extracted {len(frames)} frames")
    
    if len(frames) >= num_frames:
        indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    video_tensor = torch.stack(frames).unsqueeze(0)
    video_tensor = video_tensor.permute(0, 1, 2, 3, 4)
    
    return video_tensor


def predict_word(model, video_tensor, idx2word):
    """Run inference and get prediction"""
    with torch.no_grad():
        video_tensor = video_tensor.to(DEVICE)
        outputs = model(video_tensor)
        
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = probs.max(1)
        
        pred_word = idx2word[pred_class.item()]
        confidence_pct = confidence.item() * 100
        
        top3_probs, top3_indices = probs.topk(3, dim=1)
        top3_words = [(idx2word[idx.item()], prob.item() * 100) 
                      for idx, prob in zip(top3_indices[0], top3_probs[0])]
    
    return pred_word, confidence_pct, top3_words


def main():
    parser = argparse.ArgumentParser(description="Lip Reading Video Demo")
    parser.add_argument("--video", type=str, help="Path to video file (e.g., lrw/GREAT/test/GREAT_00001.mp4)")
    parser.add_argument("--model", type=str, default="bilstm", 
                        choices=["bilstm", "gru"],
                        help="Model to use (default: bilstm)")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" LIP READING VIDEO DEMO")
    print("="*70)
    
    if args.video:
        video_path = args.video
    else:
        print("\n📂 No video specified. Here are some examples from your test set:")
        print("\nExample commands:")
        print("  python3 demo_video_file.py --video lrw/GREAT/test/GREAT_00001.mp4")
        print("  python3 demo_video_file.py --video lrw/PUBLIC/test/PUBLIC_00001.mp4")
        print("  python3 demo_video_file.py --video lrw/PLANS/test/PLANS_00001.mp4")
        print("  python3 demo_video_file.py --video lrw/YOUNG/test/YOUNG_00001.mp4")
        print("  python3 demo_video_file.py --video lrw/KILLED/test/KILLED_00001.mp4")
        print("\nOr enter path manually:")
        video_path = input("Enter video file path: ").strip()
    
    if not video_path:
        print(" No video path provided. Exiting.")
        return
    
    print(f"\nLoading {args.model.upper()} model...")
    model, idx2word, model_name = load_model(args.model)
    print(f"Model loaded: {model_name}")
    print(f"Classes: {list(idx2word.values())}")
    
    print(f"\n Processing video...")
    video_tensor = process_video(video_path)
    
    if video_tensor is None:
        return
    
    print(f"Video processed: shape {video_tensor.shape}")
    
    print(f"\n Running inference with {model_name} model...")
    pred_word, confidence, top3 = predict_word(model, video_tensor, idx2word)
    
    print("\n" + "="*70)
    print("DETECTION RESULT")
    print("="*70)
    print(f"\n✨ Detected Word: **{pred_word}**")
    print(f" Confidence: {confidence:.2f}%")
    print(f" Video File: {os.path.basename(video_path)}")
    print(f" Model Used: {model_name}")
    
    print(f"\n Top 3 Predictions:")
    for i, (word, conf) in enumerate(top3, 1):
        bar = "█" * int(conf / 5)
        print(f"  {i}. {word:10s} {conf:6.2f}% {bar}")
    
    video_name = os.path.basename(video_path).upper()
    if pred_word in video_name:
        print(f"\nCORRECT! Video is '{pred_word}' and model predicted '{pred_word}'")
    else:
        actual_word = None
        for word in idx2word.values():
            if word in video_name:
                actual_word = word
                break
        if actual_word:
            print(f"\n Incorrect. Video is '{actual_word}' but model predicted '{pred_word}'")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
