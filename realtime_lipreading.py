import cv2
import torch
import json
import collections
import numpy as np
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Using full frame instead of lip region.")
from src.models.cnn_bilstm import CNNBiLSTM

SEQ_LEN = 15
IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("saved_models/labels.json", "r") as f:
    word2idx = json.load(f)

idx2word = {v: k for k, v in word2idx.items()}
num_classes = len(word2idx)

print(f"Loaded {num_classes} classes: {list(word2idx.keys())}")

model = CNNBiLSTM(num_classes).to(DEVICE)
model.load_state_dict(
    torch.load("saved_models/cnn_bilstm.pth", map_location=DEVICE)
)
model.eval()

if DLIB_AVAILABLE:
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print("Dlib face detector initialized")
    except:
        print("Warning: Could not load dlib shape predictor. Using full frame.")
        DLIB_AVAILABLE = False

frame_buffer = collections.deque(maxlen=SEQ_LEN)

def extract_lip_region(frame):
    """Extract lip region from frame using face landmarks"""
    if not DLIB_AVAILABLE:
        h, w = frame.shape[:2]
        y1, y2 = int(h * 0.6), int(h * 0.9)
        x1, x2 = int(w * 0.3), int(w * 0.7)
        return frame[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    faces = detector(gray)
    
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        lip_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] 
                              for i in range(48, 68)])
        
        x_min, y_min = lip_points.min(axis=0)
        x_max, y_max = lip_points.max(axis=0)
        
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(gray.shape[1], x_max + margin)
        y_max = min(gray.shape[0], y_max + margin)
        
        return gray[y_min:y_max, x_min:x_max]
    else:
        h, w = gray.shape[:2]
        y1, y2 = int(h * 0.6), int(h * 0.9)
        x1, x2 = int(w * 0.3), int(w * 0.7)
        return gray[y1:y2, x1:x2]

cap = cv2.VideoCapture(0)
print("\n=== Real-time Lip Reading Demo ===")
print(f"Expected classes: {list(word2idx.keys())}")
print(f"Using sequence length: {SEQ_LEN} frames")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    lip_region = extract_lip_region(frame)
    
    if len(lip_region.shape) == 3:
        gray = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = lip_region
    
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    tensor = torch.tensor(resized, dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0)
    frame_buffer.append(tensor)

    pred_word = "Collecting frames..."
    confidence = 0.0

    if len(frame_buffer) == SEQ_LEN:
        x = torch.stack(list(frame_buffer))
        x = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(x)
            probabilities = torch.softmax(out, dim=1)
            confidence, pred_idx = probabilities.max(1)
            pred_idx = pred_idx.item()
            confidence = confidence.item()
            pred_word = idx2word[pred_idx]
            

            top3_probs, top3_indices = probabilities.topk(3, dim=1)
            print(f"\rTop predictions: ", end="")
            for i in range(3):
                word = idx2word[top3_indices[0][i].item()]
                prob = top3_probs[0][i].item()
                print(f"{word}({prob:.2%}) ", end="")


    display_text = f"Prediction: {pred_word}"
    if len(frame_buffer) == SEQ_LEN:
        display_text += f" ({confidence:.1%})"
    
    cv2.putText(
        frame,
        display_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    
    buffer_text = f"Buffer: {len(frame_buffer)}/{SEQ_LEN}"
    cv2.putText(
        frame,
        buffer_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    cv2.imshow("Real-time Lip Reading", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
