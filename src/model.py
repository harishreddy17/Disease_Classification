import torch
from torchvision import models
from config import NUM_CLASSES, DEVICE

class OnionResNet:
    def __init__(self, num_classes=NUM_CLASSES, device=DEVICE):
        self.device = device
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(device)
    
    def forward(self, x):
        return self.model(x)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
