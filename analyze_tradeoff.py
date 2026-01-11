import torch
import time
import json
import os
from torch.utils.data import DataLoader
from src.datasets.lrw_dataset import LRWDataset
from src.models.cnn_bilstm import CNNBiLSTM
from src.models.cnn_gru import CNNGRU

DATA_DIR = "lrw"
MODEL_DIR = "saved_models"
BATCH_SIZE = 4
SEQ_LEN = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(MODEL_DIR, "labels.json"), "r") as f:
    word2idx = json.load(f)

num_classes = len(word2idx)


test_ds = LRWDataset(DATA_DIR, split="val")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def evaluate_model(model):
    model.eval()
    correct = total = 0
    infer_times = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            start = time.time()
            out = model(x)
            torch.cuda.synchronize() if DEVICE.type == "cuda" else None
            end = time.time()

            infer_times.append(end - start)

            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    avg_latency = sum(infer_times) / len(infer_times)
    fps = 1 / avg_latency

    return acc, avg_latency, fps


bilstm = CNNBiLSTM(num_classes).to(DEVICE)
gru = CNNGRU(num_classes).to(DEVICE)

bilstm.load_state_dict(torch.load(f"{MODEL_DIR}/cnn_bilstm.pth", map_location=DEVICE))
gru.load_state_dict(torch.load(f"{MODEL_DIR}/cnn_gru.pth", map_location=DEVICE))


results = {}

for name, model, path in [
    ("CNN + BiLSTM", bilstm, f"{MODEL_DIR}/cnn_bilstm.pth"),
    ("CNN + GRU", gru, f"{MODEL_DIR}/cnn_gru.pth"),
]:
    acc, latency, fps = evaluate_model(model)

    results[name] = {
        "Accuracy (%)": round(acc, 2),
        "Latency (s)": round(latency, 4),
        "FPS": round(fps, 2),
        "Parameters (M)": round(count_parameters(model) / 1e6, 2),
        "Model Size (MB)": round(model_size_mb(path), 2),
    }


print("\nMODEL COMPARISON RESULTS\n")
for model_name, metrics in results.items():
    print(f"{model_name}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("-" * 40)
