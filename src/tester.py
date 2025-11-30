import torch
from config import DEVICE, CLASS_NAMES
from src.logger import logger

class Tester:
    def __init__(self, model, test_loader):
        self.model = model.model
        self.test_loader = test_loader
        self.device = DEVICE

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        class_correct = [0]*len(CLASS_NAMES)
        class_total = [0]*len(CLASS_NAMES)
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i in range(len(labels)):
                    label = labels[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        logger.info(f"Overall Test Accuracy: {100*correct/total:.2f}%")
        for i, class_name in enumerate(CLASS_NAMES):
            if class_total[i] > 0:
                logger.info(f"{class_name}: {100*class_correct[i]/class_total[i]:.2f}%")
