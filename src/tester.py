# src/tester.py
import torch
from .config import BaseConfig
from src.logger import logger

class Tester:
    def __init__(self, model, test_loader, config: BaseConfig):
        """
        model: OnionClassifier instance
        test_loader: PyTorch DataLoader
        config: BaseConfig instance
        """
        self.model = model.model
        self.test_loader = test_loader
        self.config = config
        self.device = config.device

    def evaluate(self):
        """Evaluate the model on test dataset and log per-class accuracy"""
        self.model.eval()
        correct, total = 0, 0
        class_correct = [0] * self.config.num_classes
        class_total = [0] * self.config.num_classes

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

        overall_acc = 100 * correct / total if total > 0 else 0
        logger.info(f"Overall Test Accuracy: {overall_acc:.2f}%")

        for i, class_name in enumerate(self.config.class_names):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                logger.info(f"{class_name}: {acc:.2f}%")
