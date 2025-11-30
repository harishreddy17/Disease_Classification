from src.dataset import OnionDataset
from src.model import OnionResNet
from src.trainer import Trainer
from src.tester import Tester
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from config import DEVICE, BATCH_SIZE
from src.logger import logger

# Dataset paths
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# Initialize dataset
dataset = OnionDataset(train_dir, val_dir)
train_loader, val_loader = dataset.get_loaders()

test_dataset = ImageFolder(test_dir, transform=dataset.val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model_wrapper = OnionResNet()

# Train
trainer = Trainer(model_wrapper, train_loader, val_loader)
trainer.train()

# Save
model_wrapper.save("resnet_onion.pth")
logger.info("Model saved to resnet_onion.pth")

# Test
tester = Tester(model_wrapper, test_loader)
tester.evaluate()
