# main.py
import os
from src.config import load_config
from src.dataset import OnionDataset
from src.model import OnionClassifier
from src.trainer import Trainer
from src.tester import Tester
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.logger import logger


def main():
    # -----------------------------
    # Load configuration
    # -----------------------------
    cfg = load_config()
    raw_dataset_dir = r"C:\Users\haris\OneDrive\Documents\image_classification\dataset"

    # -----------------------------
    # Initialize dataset
    # -----------------------------
    print("✅ Splitting dataset into train/val/test...")
    dataset = OnionDataset(
        raw_dir=raw_dataset_dir,
        config=cfg,
        augment=True,
        split=False  # split into train/val/test automatically
    )
    train_loader, val_loader, test_loader = dataset.get_loaders()
    print("✅ Dataset loaders ready.")

    # -----------------------------
    # Initialize model
    # -----------------------------
    model_wrapper = OnionClassifier(cfg)
    model_wrapper.model.to(cfg.device)

    # -----------------------------
    # Training
    # -----------------------------
    trainer = Trainer(model_wrapper, train_loader, val_loader, cfg)
    trainer.train()  # automatically saves the best model

    # -----------------------------
    # Testing
    # -----------------------------
    tester = Tester(model_wrapper, test_loader, cfg)
    tester.evaluate()


if __name__ == "__main__":
    main()