# src/dataset.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.logger import logger

class OnionDataset:
    def __init__(self, raw_dir, config, augment=True, split=True):
        self.raw_dir = raw_dir
        self.config = config
        self.augment = augment

        self.root = raw_dir  # this is your dataset folder
        self.train_dir = os.path.join(self.root, "train")
        self.val_dir = os.path.join(self.root, "val")
        self.test_dir = os.path.join(self.root, "test")

        # If first run, perform split
        if split:
            self.split_dataset()

        # Set transforms
        self.train_transform = self.build_train_transform()
        self.val_transform = self.build_val_transform()

    # -----------------------------------------------------
    # 1. **Dataset Split**
    # -----------------------------------------------------
    def split_dataset(self):
        """Split raw dataset into train/val/test inside processed_data/."""
        import shutil
        from sklearn.model_selection import train_test_split

        if os.path.exists(self.train_dir) and os.listdir(self.train_dir):
            logger.info("Train/Val/Test already exists. Skipping split.")
            return

        logger.info("Splitting dataset into train/val/test...")

        for class_name in os.listdir(self.raw_dir):
            class_path = os.path.join(self.raw_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = [
                os.path.join(class_path, img)
                for img in os.listdir(class_path)
                if img.lower().endswith((".jpg", ".png", ".jpeg"))
            ]

            train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
            train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.2, random_state=42)

            for split_name, img_list in [
                ("train", train_imgs),
                ("val", val_imgs),
                ("test", test_imgs)
            ]:
                split_dir = os.path.join(self.processed_dir, split_name, class_name)
                os.makedirs(split_dir, exist_ok=True)

                for img in img_list:
                    shutil.copy(img, os.path.join(split_dir, os.path.basename(img)))

        logger.info("Dataset split completed.")

    # -----------------------------------------------------
    # 2. **Transforms**
    # -----------------------------------------------------
    def build_train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])

    def build_val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
        ])

    # -----------------------------------------------------
    # 3. **Return DataLoaders**
    # -----------------------------------------------------
    def get_loaders(self):
        train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        print("Classes:", train_dataset.classes)
        print("Number of classes:", len(train_dataset.classes))
        val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_transform)
        test_dataset = datasets.ImageFolder(self.test_dir, transform=self.val_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

        return train_loader, val_loader, test_loader
