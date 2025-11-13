import os
import csv
import random
from pathlib import Path
from typing import Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Configuration defaults

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



# list files by class

def _scan_class_folders(base_dir: str, classes: List[str]) -> List[Tuple[str, int]]:
    """
    Returns list of (filepath, label_idx)
    Expects structure:
      base_dir/
        train/
          open/
          close/
        test/   (optional, not used for splitting here)
    """
    items = []
    for idx, cls in enumerate(classes):
        cls_folder = Path(base_dir) / "train" / cls
        if not cls_folder.exists():
            raise FileNotFoundError(f"Expected folder: {cls_folder}")
        for p in cls_folder.glob("**/*.png"):
            items.append((str(p.resolve()), idx))
    return items



# per-class splitting

def stratified_split(items: List[Tuple[str,int]], val_pct: float, test_pct: float, seed: int=42):
    """
    Splits items into train/val/test maintaining class ratios.
    items: list of (path, label)
    returns: dict with keys "train","val","test" mapping to lists of (path,label)
    """
    random.seed(seed)
    by_class = {}
    for p,l in items:
        by_class.setdefault(l, []).append(p)

    train_list, val_list, test_list = [], [], []
    for label, paths in by_class.items():
        random.shuffle(paths)
        n = len(paths)
        n_test = int(round(n * test_pct))
        n_val  = int(round(n * val_pct))
        if n_test + n_val >= n:
            n_test = max(0, min(n_test, n-2))
            n_val  = max(0, min(n_val, n-1-n_test))

        test_paths = paths[:n_test]
        val_paths  = paths[n_test:n_test + n_val]
        train_paths= paths[n_test + n_val:]

        test_list.extend([(p, label) for p in test_paths])
        val_list.extend([(p, label) for p in val_paths])
        train_list.extend([(p, label) for p in train_paths])

    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    return {"train": train_list, "val": val_list, "test": test_list}


# from file list
class FileListImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str,int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# Transforms
def get_transforms(image_size: int = 224):
    """
    Returns (train_transform, val_transform, test_transform)
    Uses ImageNet normalization by default.
    """
    train_t = transforms.Compose([
        transforms.Resize((int(image_size*1.1), int(image_size*1.1))),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_t, val_test_t, val_test_t



# CSV manifest writer
def _write_manifest(manifest_path: str, items: List[Tuple[str,int]]):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        for p,l in items:
            writer.writerow([p, l])


# Main entry: dataloaders
def get_dataloaders(
    data_dir: str = "data/raw",
    classes: List[str] = ["open", "close"],
    image_size: int = 80,
    batch_size: int = 32,
    val_pct: float = 0.1,
    test_pct: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    save_manifests: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build and return (train_loader, val_loader, test_loader, meta)
    meta contains counts and class mapping.
    """
    items = _scan_class_folders(data_dir, classes)

    # split 
    splits = stratified_split(items, val_pct=val_pct, test_pct=test_pct, seed=seed)

    # transforms
    train_tf, val_tf, test_tf = get_transforms(image_size=image_size)

    # datasets
    ds_train = FileListImageDataset(splits["train"], transform=train_tf)
    ds_val   = FileListImageDataset(splits["val"], transform=val_tf)
    ds_test  = FileListImageDataset(splits["test"], transform=test_tf)

    # dataloaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # manifests
    if save_manifests:
        manifest_root = Path("manifests")
        _write_manifest(str(manifest_root / "train.csv"), splits["train"])
        _write_manifest(str(manifest_root / "val.csv"),   splits["val"])
        _write_manifest(str(manifest_root / "test.csv"),  splits["test"])

    # meta
    meta = {
        "num_train": len(splits["train"]),
        "num_val":   len(splits["val"]),
        "num_test":  len(splits["test"]),
        "classes": classes,
        "class_to_idx": {c:i for i,c in enumerate(classes)}
    }

    return dl_train, dl_val, dl_test, meta

# CLI-run support
if __name__ == "__main__":
    # quick run to create manifest files and print dataset sizes
    dl_train, dl_val, dl_test, meta = get_dataloaders(
        data_dir="data/raw",
        classes=["open", "close"],
        image_size=224,
        batch_size=32,
        val_pct=0.1,
        test_pct=0.1,
        num_workers=2,
        save_manifests=True,
        seed=42,
    )
    print("Data loaders created.")
    print(f"Train samples: {meta['num_train']}")
    print(f"Val   samples: {meta['num_val']}")
    print(f"Test  samples: {meta['num_test']}")
    print("Manifests written to ./manifests/*.csv")
