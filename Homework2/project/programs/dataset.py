from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

SVHN_TRAIN_FILE = "train_32x32.mat"
SVHN_TEST_FILE = "test_32x32.mat"
SVHN_TRAIN_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
SVHN_TEST_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


class SVHNMatDataset(Dataset):
    """SVHN Format 2 (.mat) loader with label remapping (10 -> 0)."""

    def __init__(self, mat_path: Path, transform: Optional[Callable] = None) -> None:
        self.mat_path = Path(mat_path)
        self.transform = transform

        if not self.mat_path.exists():
            raise FileNotFoundError(f"SVHN file not found: {self.mat_path}")

        mat = loadmat(str(self.mat_path))
        if "X" not in mat or "y" not in mat:
            raise KeyError(f"{self.mat_path.name} missing required keys: X / y")

        images = mat["X"]
        labels = mat["y"]

        if images.ndim != 4 or images.shape[:3] != (32, 32, 3):
            raise ValueError(
                f"Unexpected X shape in {self.mat_path.name}: {images.shape}, expected (32, 32, 3, N)"
            )

        self.images = np.transpose(images, (3, 0, 1, 2)).astype(np.uint8)
        self.labels = labels.reshape(-1).astype(np.int64)
        self.labels[self.labels == 10] = 0

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = int(self.labels[idx])

        pil_image = Image.fromarray(image)
        if self.transform is None:
            tensor = transforms.ToTensor()(pil_image)
        else:
            tensor = self.transform(pil_image)

        return tensor, label


@dataclass
class SVHNDataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    test_size: int
    num_workers: int
    train_path: Path
    test_path: Path


def default_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD),
        ]
    )


def default_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=SVHN_MEAN, std=SVHN_STD),
        ]
    )


def infer_num_workers(requested_workers: int) -> int:
    if requested_workers >= 0:
        return requested_workers

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    return min(8, cpu_count - 1)


def ensure_svhn_files(resource_dir: Path, download_if_missing: bool = False) -> tuple[Path, Path]:
    resource_dir = Path(resource_dir)
    resource_dir.mkdir(parents=True, exist_ok=True)

    train_path = resource_dir / SVHN_TRAIN_FILE
    test_path = resource_dir / SVHN_TEST_FILE

    missing_files = [p for p in [train_path, test_path] if not p.exists()]
    if not missing_files:
        return train_path, test_path

    if not download_if_missing:
        missing_names = ", ".join(p.name for p in missing_files)
        raise FileNotFoundError(
            "SVHN source files are missing in resource directory. "
            f"Missing: {missing_names}. "
            "Place them manually or set --download-if-missing."
        )

    if not train_path.exists():
        urllib.request.urlretrieve(SVHN_TRAIN_URL, train_path)
    if not test_path.exists():
        urllib.request.urlretrieve(SVHN_TEST_URL, test_path)

    return train_path, test_path


def _maybe_limit_dataset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None or limit <= 0:
        return dataset
    if limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def build_dataloaders(
    resource_dir: Path,
    batch_size: int,
    requested_workers: int,
    pin_memory: bool,
    download_if_missing: bool = False,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
) -> SVHNDataBundle:
    train_path, test_path = ensure_svhn_files(resource_dir=resource_dir, download_if_missing=download_if_missing)

    train_dataset: Dataset = SVHNMatDataset(train_path, transform=default_train_transform())
    test_dataset: Dataset = SVHNMatDataset(test_path, transform=default_test_transform())

    train_dataset = _maybe_limit_dataset(train_dataset, train_limit)
    test_dataset = _maybe_limit_dataset(test_dataset, test_limit)

    num_workers = infer_num_workers(requested_workers)

    loader_common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_common)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_common)

    return SVHNDataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        train_size=len(train_dataset),
        test_size=len(test_dataset),
        num_workers=num_workers,
        train_path=train_path,
        test_path=test_path,
    )


def validate_batch_shape_and_labels(
    loader: DataLoader,
    num_classes: int = 10,
    channels: int = 3,
    image_size: int = 32,
) -> dict:
    images, labels = next(iter(loader))

    if images.ndim != 4:
        raise AssertionError(f"Expected image ndim=4, got {images.ndim}")
    if images.shape[1] != channels or images.shape[2] != image_size or images.shape[3] != image_size:
        raise AssertionError(f"Unexpected image batch shape: {tuple(images.shape)}")

    label_min = int(labels.min().item())
    label_max = int(labels.max().item())

    if label_min < 0 or label_max >= num_classes:
        raise AssertionError(f"Label out of range: min={label_min}, max={label_max}")

    return {
        "batch_shape": [int(v) for v in images.shape],
        "label_min": label_min,
        "label_max": label_max,
        "dtype": str(images.dtype),
    }
