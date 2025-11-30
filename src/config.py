import torch
import os

NUM_CLASSES = 5
CLASS_NAMES = ["healthy", "downy_mildew", "purple_blotch", "botrytis", "smut"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
MODEL_PATH = "resnet_onion.pth"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
