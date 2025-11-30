from PIL import Image
import torch
from torchvision import transforms
from config import IMG_SIZE, DEVICE, CLASS_NAMES
from src.model import OnionResNet
from src.logger import logger
import time
import psutil

class Predictor:
    def __init__(self, model_path):
        self.device = DEVICE
        self.model_wrapper = OnionResNet()
        self.model_wrapper.load(model_path)
        self.model_wrapper.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    
    def preprocess(self, image_file):
        image = Image.open(image_file).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image_tensor):
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model_wrapper.forward(image_tensor)
            _, predicted = torch.max(outputs,1)
        class_name = CLASS_NAMES[predicted.item()]
        elapsed = time.time()-start_time
        # Log GPU/CPU usage
        gpu_mem = torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        logger.info(f"Prediction: {class_name}, Time: {elapsed:.4f}s, CPU: {cpu}%, MEM: {mem}%, GPU_MB: {gpu_mem:.2f}")
        return class_name
