from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import IMG_SIZE, BATCH_SIZE

class OnionDataset:
    def __init__(self, train_dir, val_dir):
        self.train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.train_dir = train_dir
        self.val_dir = val_dir

    def get_loaders(self):
        train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, val_loader
