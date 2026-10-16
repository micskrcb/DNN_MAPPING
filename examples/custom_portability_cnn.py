"""Small example for --custom_model portability checks."""
import torch
from torch import nn


class PortableCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU())
        self.classifier = nn.Linear(16 * 16 * 16, 10)

    def forward(self, inputs):
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


def build_model():
    return PortableCNN(), torch.randn(1, 3, 32, 32)
