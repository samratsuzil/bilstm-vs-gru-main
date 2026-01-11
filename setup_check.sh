echo "========================================"
echo "Lip Reading System - Diagnostic & Setup"
echo "========================================"
echo ""

echo "1. Checking Python installation..."
python --version
if [ $? -ne 0 ]; then
    echo "Python 3 not found. Please install Python 3.8+"
    exit 1
fi
echo "Python OK"
echo ""

echo "2. Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PyTorch not found. Installing..."
    pip3 install torch torchvision
fi
echo "PyTorch OK"
echo ""

echo "3. Checking OpenCV..."
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "OpenCV not found. Installing..."
    pip3 install opencv-python
fi
echo "OpenCV OK"
echo ""

echo "4. Checking dataset structure..."
if [ ! -d "lrw" ]; then
    echo "Dataset directory 'lrw' not found!"
    echo "Please ensure your dataset is in the correct location."
    exit 1
fi

num_classes=$(ls -d lrw/*/ 2>/dev/null | wc -l)
echo "Found $num_classes word classes"

for class_dir in lrw/*/; do
    class_name=$(basename "$class_dir")
    if [ ! -d "$class_dir/train" ] || [ ! -d "$class_dir/val" ] || [ ! -d "$class_dir/test" ]; then
        echo "Warning: $class_name missing train/val/test splits"
    fi
done
echo ""

echo "5. Checking saved models..."
if [ -f "saved_models/cnn_bilstm.pth" ]; then
    size=$(du -h saved_models/cnn_bilstm.pth | cut -f1)
    echo "BiLSTM model found ($size)"
else
    echo "BiLSTM model not found. You need to train it."
fi

if [ -f "saved_models/cnn_gru.pth" ]; then
    size=$(du -h saved_models/cnn_gru.pth | cut -f1)
    echo "GRU model found ($size)"
else
    echo "GRU model not found. You need to train it."
fi

if [ -f "saved_models/labels.json" ]; then
    echo "Labels file found"
    echo "  Classes:"
    python3 -c "import json; labels = json.load(open('saved_models/labels.json')); print('  ', list(labels.keys()))"
else
    echo "Labels file not found"
fi
echo ""

echo "6. Diagnosing common issues..."



if [ "$train_seq" != "$realtime_seq" ]; then
    echo "WARNING: Sequence length mismatch detected!"
    echo "   Training uses: $train_seq frames"
    echo "   Inference uses: $realtime_seq frames"
    echo "   These MUST match! This causes incorrect predictions."
else
    echo "Sequence lengths match ($train_seq frames)"
fi
echo ""

echo "7. Checking dlib (optional but recommended)..."
python3 -c "import dlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "dlib not installed. Lip detection will use fallback method."
    echo "   To install: pip3 install dlib"
    echo "   (May require cmake: sudo apt-get install cmake)"
else
    echo "dlib installed"
    
    if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
        echo "Face landmark model not found"
        echo "   Download with:"
        echo "   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        echo "   bunzip2 shape_predictor_68_face_landmarks.dat.bz2"
    else
        echo "Face landmark model found"
    fi
fi
echo ""

echo "========================================"
echo "GPU Information"
echo "========================================"
echo ""

python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')" 2>/dev/null

echo ""
echo "========================================"
echo "Setup check complete!"
echo "========================================"
