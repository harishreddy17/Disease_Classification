# src/preprocessor.py
import os
import shutil
import random
import hashlib
from collections import defaultdict
from pathlib import Path
from PIL import Image
import cv2
from torchvision import transforms

class Preprocessor:
    def __init__(self, config, raw_dir, processed_dir, val_ratio=0.2, test_ratio=0.1, crop=True):
        self.config = config
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.crop = crop

        # Output directories
        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "val"
        self.test_dir = self.processed_dir / "test"
        for d in [self.train_dir, self.val_dir, self.test_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 1. Detect Duplicates
    # ----------------------------
    def find_duplicates(self):
        hashes = defaultdict(list)
        duplicates = []
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.lower().endswith((".png",".jpg",".jpeg")):
                    path = Path(root)/file
                    file_hash = hashlib.md5(path.read_bytes()).hexdigest()
                    if file_hash in hashes:
                        duplicates.append(path)
                    hashes[file_hash].append(path)
        return duplicates

    # ----------------------------
    # 2. Blurry Detection
    # ----------------------------
    def is_blurry(self, path, threshold=100):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        return cv2.Laplacian(img, cv2.CV_64F).var() < threshold

    # ----------------------------
    # 3. Optional Auto-Crop
    # ----------------------------
    def auto_crop(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]

    # ----------------------------
    # 4. Dataset split
    # ----------------------------
    def split_dataset(self):
        print("✅ Splitting dataset into train/val/test...")
        for cls_folder in self.raw_dir.iterdir():
            if cls_folder.is_dir():
                images = [p for p in cls_folder.glob("*") if p.suffix.lower() in [".png",".jpg",".jpeg"]]
                random.shuffle(images)
                n = len(images)
                n_val = int(n * self.val_ratio)
                n_test = int(n * self.test_ratio)
                n_train = n - n_val - n_test

                folders = [
                    (self.train_dir/cls_folder.name, images[:n_train]),
                    (self.val_dir/cls_folder.name, images[n_train:n_train+n_val]),
                    (self.test_dir/cls_folder.name, images[n_train+n_val:])
                ]
                for target_dir, imgs in folders:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for img in imgs:
                        shutil.copy(img, target_dir/img.name)

        print("✅ Dataset split completed.")

    # ----------------------------
    # 5. Augmentations (training only)
    # ----------------------------
    def augmentations(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomResizedCrop(self.config.img_size, scale=(0.8,1.0))
        ])

    # ----------------------------
    # 6. Model-specific transforms
    # ----------------------------
    def model_preprocessing(self):
        model_name = self.config.model_name.lower()
        normalize = {
            "resnet": ([0.485,0.456,0.406],[0.229,0.224,0.225]),
            "mobilenet": ([0.485,0.456,0.406],[0.229,0.224,0.225]),
            "efficientnet": ([0.485,0.456,0.406],[0.229,0.224,0.225]),
            "vit": ([0.5,0.5,0.5],[0.5,0.5,0.5])
        }
        mean,std = normalize.get(model_name, ([0.5,0.5,0.5],[0.5,0.5,0.5]))
        return transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # ----------------------------
    # 7. Full preprocessing runner
    # ----------------------------
    def run_full_preprocessing(self):
        # Remove duplicates
        duplicates = self.find_duplicates()
        for img in duplicates:
            img.unlink()
        # Remove blurry
        for img in self.raw_dir.glob("**/*"):
            if img.suffix.lower() in [".png",".jpg",".jpeg"] and self.is_blurry(img):
                img.unlink()
        # Auto-crop
        if self.crop:
            for img in self.raw_dir.glob("**/*"):
                if img.suffix.lower() in [".png",".jpg",".jpeg"]:
                    image = cv2.imread(str(img))
                    cropped = self.auto_crop(image)
                    cv2.imwrite(str(img), cropped)
        # Split dataset
        self.split_dataset()
