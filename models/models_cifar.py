import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARImputationModel(nn.Module):
    """
    Modified PI-NAIM model for CIFAR datasets
    """

    def __init__(self, input_channels=3, hidden_dim=128, num_paths=8):
        super(CIFARImputationModel, self).__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_paths = num_paths

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Path-integrated decoder
        self.path_weights = nn.Parameter(torch.ones(num_paths) / num_paths)

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, input_channels, 3, padding=1),
                nn.Sigmoid()
            ) for _ in range(num_paths)
        ])

    def forward(self, x, mask):
        batch_size = x.shape[0]

        # Encode
        encoded = self.encoder(x)

        # Multiple decoding paths
        outputs = []
        for decoder in self.decoders:
            decoded = decoder(encoded)
            outputs.append(decoded)

        # Weighted combination
        outputs = torch.stack(outputs, dim=1)  # [B, num_paths, C, H, W]
        weights = F.softmax(self.path_weights, dim=0).view(1, -1, 1, 1, 1)
        combined = (outputs * weights).sum(dim=1)

        # Impute missing values
        imputed = x * mask + combined * (1 - mask)

        return imputed, combined


class CIFARClassifier(nn.Module):
    """
    Classifier for CIFAR to evaluate imputation quality
    """

    def __init__(self, num_classes=10):
        super(CIFARClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x