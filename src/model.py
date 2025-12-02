# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from .config import BaseConfig

class ModelFactory:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.model_name = config.model_name.lower()
        self.num_classes = config.num_classes
        self.device = config.device

    def build(self):
        if "resnet" in self.model_name:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            if self.config.freeze_base:
                for p in model.parameters():
                    p.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif "mobilenet" in self.model_name:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            if self.config.freeze_base:
                for p in model.parameters():
                    p.requires_grad = False
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)

        elif "efficientnet" in self.model_name:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            if self.config.freeze_base:
                for p in model.parameters():
                    p.requires_grad = False
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        elif "vit" in self.model_name:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            if self.config.freeze_base:
                for p in model.parameters():
                    p.requires_grad = False
            model.heads.head = nn.Linear(model.heads.head.in_features, self.num_classes)

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model.to(self.device)


class OnionClassifier:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.model = ModelFactory(config).build()
        self.device = config.device

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x.to(self.device))

    def save(self):
        torch.save(self.model.state_dict(), self.config.model_path())

    def load(self):
        self.model.load_state_dict(torch.load(self.config.model_path(), map_location=self.device))
        self.model.eval()
