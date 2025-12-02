# src/config.py
import os
import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BaseConfig:
    # Model selection
    model_name: str = "resnet18"
    freeze_base: bool = True
    augmentation: bool = True
    preprocessing_mode: str = "default"

    # Classes
    num_classes: int = 9  # <--- Update here
    class_names: list = ("Cardboard", "Food Organics", "Glass", "Metal",
                         "Miscellaneous Trash", "Paper", "Plastic", "Textile Trash", "Vegetation")
    num_workers: int = 4

    # Image / preprocessing
    img_size: int = 224

    # Training
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    scheduler: bool = False

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: str = None
    model_dir: str = None
    log_dir: str = None

    # Device
    device: torch.device = None

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = str(self.project_root / "dataset")
        if self.model_dir is None:
            self.model_dir = str(self.project_root / "models")
        if self.log_dir is None:
            self.log_dir = str(self.project_root / "logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # helper method to get model path dynamically
    def model_path(self):
        return os.path.join(self.model_dir, f"{self.model_name}.pth")


def load_config(env: str = "dev", **overrides) -> BaseConfig:
    if env == "dev":
        cfg = BaseConfig()
    elif env == "prod":
        cfg = BaseConfig(epochs=25, batch_size=32)
    else:
        cfg = BaseConfig()

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg
