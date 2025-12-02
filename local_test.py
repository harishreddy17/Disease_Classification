# local_test.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from config import load_config, CLASS_NAMES
from src.preprocessor import Preprocessor
from src.predictor import Predictor
from src.logger import logger
import time
import psutil

# ----------------------------
# Load config dynamically
# ----------------------------
config = load_config(model_name="resnet18")  # Change model easily
test_dir = f"{config.data_dir}/test"

# ----------------------------
# Initialize Preprocessor
# ----------------------------
preprocessor = Preprocessor(config)
test_transform = preprocessor.model_preprocessing()

# ----------------------------
# Create test dataset and loader
# ----------------------------
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ----------------------------
# Initialize Predictor
# ----------------------------
predictor = Predictor(model_path=f"{config.model_dir}/resnet_onion.pth", config=config)

# ----------------------------
# Run evaluation
# ----------------------------
total, correct = 0, 0
class_correct = [0]*len(CLASS_NAMES)
class_total = [0]*len(CLASS_NAMES)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(config.device), labels.to(config.device)
        start_time = time.time()
        
        outputs = predictor.model_wrapper.forward(images)
        _, predicted = torch.max(outputs, 1)
        
        elapsed = time.time() - start_time
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        gpu_mem = torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0
        
        logger.info(f"Prediction Time: {elapsed:.4f}s, CPU: {cpu}%, MEM: {mem}%, GPU_MB: {gpu_mem:.2f}")
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            class_total[labels[i]] += 1
            if predicted[i] == labels[i]:
                class_correct[labels[i]] += 1

# ----------------------------
# Print overall and class-wise accuracy
# ----------------------------
overall_acc = 100 * correct / total
print(f"\nOverall Test Accuracy: {overall_acc:.2f}%\n")

print("Class-wise Accuracy:")
for i, class_name in enumerate(CLASS_NAMES):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{class_name}: {acc:.2f}%")
