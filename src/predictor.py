# src/predictor.py
from PIL import Image
import torch
from src.model import OnionClassifier
from src.preprocessor import Preprocessor
from src.logger import logger
import time
import psutil

class Predictor:
    def __init__(self, model_path: str, config):
        self.config = config
        self.device = config.device

        # Load model
        self.model_wrapper = OnionClassifier(config)
        self.model_wrapper.load()

        # Preprocessing pipeline from Preprocessor
        self.preprocessor = Preprocessor(config)
        self.transform = self.preprocessor.model_preprocessing()

    def preprocess(self, image_file):
        """Preprocess a single image for prediction"""
        image = Image.open(image_file).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, image_tensor):
        """Predict class and log resources"""
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model_wrapper.forward(image_tensor)
            _, predicted = torch.max(outputs, 1)

        class_name = self.config.class_names[predicted.item()]
        elapsed = time.time() - start_time

        # Resource logging
        gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        logger.info(f"Prediction: {class_name}, Time: {elapsed:.4f}s, CPU: {cpu}%, MEM: {mem}%, GPU_MB: {gpu_mem:.2f}")
        return class_name
