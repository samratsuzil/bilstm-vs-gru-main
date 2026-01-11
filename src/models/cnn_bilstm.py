import torch
import torch.nn as nn
from .cnn_backbone import CNNBackbone

class AttentionLayer(nn.Module):
    """
    Self-attention mechanism for temporal feature aggregation
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs):
        # rnn_outputs: (B, T, hidden_dim)
        attention_weights = torch.softmax(self.attention(rnn_outputs), dim=1)
        # attention_weights: (B, T, 1)
        attended = torch.sum(attention_weights * rnn_outputs, dim=1)
        # attended: (B, hidden_dim)
        return attended, attention_weights

class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes, dropout=0.5, use_resnet50=False, hidden_size=512, num_layers=3):
        """
        Enhanced BiLSTM model for 85%+ accuracy
        Args:
            num_classes: Number of output classes
            dropout: Dropout rate
            use_resnet50: Use ResNet50 instead of ResNet18
            hidden_size: LSTM hidden size (default 512 for more capacity)
            num_layers: Number of LSTM layers (default 3 for deeper model)
        """
        super().__init__()
        self.cnn = CNNBackbone(use_resnet50=use_resnet50)
        cnn_output_dim = self.cnn.output_dim

        # Powerful BiLSTM with more layers and larger hidden size
        self.rnn = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Bidirectional doubles the hidden size
        rnn_output_dim = hidden_size * 2

        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(rnn_output_dim)

        # Attention mechanism for better temporal aggregation
        self.attention = AttentionLayer(rnn_output_dim)

        # Optimized classifier with fewer parameters for better speed
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_output_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(256, num_classes)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Proper weight initialization for better convergence
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # (B, T, cnn_output_dim)

        # BiLSTM processing
        x, _ = self.rnn(x)  # (B, T, rnn_output_dim)

        # Layer normalization
        x = self.layer_norm(x)

        # Attention-based temporal aggregation
        x, _ = self.attention(x)  # (B, rnn_output_dim)

        # Optimized classifier with batch normalization
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dropout(x)
        x = self.fc2(x)

        return x