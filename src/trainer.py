import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE, EPOCHS, LEARNING_RATE
from src.logger import logger

class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=LEARNING_RATE)
        self.device = DEVICE

    def train(self, epochs=EPOCHS):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(self.train_loader)
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            self.validate()

    def validate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100*correct/total
        logger.info(f"Validation Accuracy: {acc:.2f}%")
