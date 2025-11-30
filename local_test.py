import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import DEVICE, IMG_SIZE, BATCH_SIZE, CLASS_NAMES
from src.predictor import Predictor
from src.logger import logger
import time
import psutil
import os

# Paths
test_dir = "dataset/test"  # replace with your test dataset path

# Transform
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load test dataset
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
predictor = Predictor(model_path="resnet_onion.pth")

# Tracking
total = 0
correct = 0
class_correct = [0]*len(CLASS_NAMES)
class_total = [0]*len(CLASS_NAMES)

print("Starting local test...")

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        start_time = time.time()
        outputs = predictor.model_wrapper.forward(images)
        _, predicted = torch.max(outputs, 1)
        elapsed = time.time()-start_time
        
        # CPU & GPU usage
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        gpu_mem = torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0
        
        logger.info(f"Prediction Time: {elapsed:.4f}s, CPU: {cpu}%, MEM: {mem}%, GPU_MB: {gpu_mem:.2f}")
        
        # Accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            class_total[labels[i]] += 1
            if predicted[i] == labels[i]:
                class_correct[labels[i]] += 1

# Overall Accuracy
overall_acc = 100 * correct / total
print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")

# Class-wise Accuracy
print("Class-wise Accuracy:")
for i, class_name in enumerate(CLASS_NAMES):
    if class_total[i] > 0:
        acc = 100*class_correct[i]/class_total[i]
        print(f"{class_name}: {acc:.2f}%")
