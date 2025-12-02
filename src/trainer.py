# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from .config import BaseConfig
from src.logger import logger
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, config: BaseConfig):
        self.model = model.model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.get_optimizer()
        self.scheduler = None
        if getattr(config, "scheduler", False):
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def get_optimizer(self):
        # Automatically detect trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.Adam(params, lr=self.config.lr)

    def train(self):
        best_acc = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0

            loop = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.config.epochs}] Training")
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(self.train_loader)
            logger.info(f"Epoch [{epoch+1}/{self.config.epochs}], Loss: {avg_loss:.4f}")

            val_acc = self.validate()
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.config.model_path())
                logger.info(f"Best model saved with accuracy: {best_acc:.2f}%")

            if self.scheduler:
                self.scheduler.step()

    def validate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validation")
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loop.set_postfix(val_acc=100*correct/total)
        acc = 100*correct/total
        logger.info(f"Validation Accuracy: {acc:.2f}%")
        return acc
