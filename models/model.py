import torch
import torch.nn as nn
import torch.nn.functional as F


class NAIM(nn.Module):
    """
    Original NAIM model from the PI-NAIM repository
    """

    def __init__(self, input_size=784, hidden_dim=128, num_paths=8):
        super(NAIM, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_paths = num_paths

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )

        # Path-integrated decoder
        self.path_weights = nn.Parameter(torch.ones(num_paths) / num_paths)

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, input_size),
                nn.Sigmoid()
            ) for _ in range(num_paths)
        ])

    def forward(self, x, mask):
        batch_size = x.shape[0]

        # Encode
        encoded = self.encoder(x.view(batch_size, -1))

        # Multiple decoding paths
        outputs = []
        for decoder in self.decoders:
            decoded = decoder(encoded)
            outputs.append(decoded.view(batch_size, 1, 28, 28))  # Reshape for MNIST

        # Weighted combination
        outputs = torch.stack(outputs, dim=1)  # [B, num_paths, 1, 28, 28]
        weights = F.softmax(self.path_weights, dim=0).view(1, -1, 1, 1, 1)
        combined = (outputs * weights).sum(dim=1)

        # Impute missing values
        imputed = x * mask + combined * (1 - mask)

        return imputed, combined


class PathIntegratedModel(nn.Module):
    """
    Alternative path-integrated model
    """

    def __init__(self, input_channels=1, hidden_dim=128, num_paths=8):
        super(PathIntegratedModel, self).__init__()
        # Add your PathIntegratedModel implementation here
        pass

    def forward(self, x, mask):
        # Implementation
        return x, x