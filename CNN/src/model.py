

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_model(num_classes=2):
    model = models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
