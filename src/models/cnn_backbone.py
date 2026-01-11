import torch.nn as nn
from torchvision import models

class CNNBackbone(nn.Module):
    def __init__(self, use_resnet50=False):
        """
        Enhanced CNN backbone with option for deeper ResNet
        Args:
            use_resnet50: If True, use ResNet50 instead of ResNet18 for better feature extraction
        """
        super().__init__()
        if use_resnet50:
            resnet = models.resnet50(pretrained=True)
        else:
            resnet = models.resnet18(pretrained=True)

        # Modify first conv layer to accept grayscale input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Extract all layers except final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Output dimension: 512 for ResNet18, 2048 for ResNet50
        self.output_dim = 2048 if use_resnet50 else 512

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.features(x).squeeze(-1).squeeze(-1)
        return x.view(B, T, -1)
