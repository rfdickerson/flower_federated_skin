import os
from typing import Tuple, Dict

import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.manual_seed(17)


class ImageFolder(Dataset):
    """Custom implementation of ImageFolder to work with Albumentations"""
    def __init__(self, root_dir, transform=None, total_classes=None):
        self.transform = transform
        self.data = []

        if total_classes:
            self.classnames = os.listdir(root_dir)[:total_classes]  # for test
        else:
            self.classnames = os.listdir(root_dir)

        for index, label in enumerate(self.classnames):
            root_image_name = os.path.join(root_dir, label)

            for i in os.listdir(root_image_name):
                full_path = os.path.join(root_image_name, i)
                self.data.append((full_path, index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        full_path, target = self.data[index]

        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, target


def load_data(dataset_dir, batch_size=32) -> Tuple[
    torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict
]:

    transform = A.Compose([
        A.HorizontalFlip(),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

    train_dataset = ImageFolder(os.path.join(dataset_dir, 'train'), transform)

    val_dataset = ImageFolder(os.path.join(dataset_dir, 'val'), transform)

    # test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transforms.Compose([
    #     transforms.RandomRotation(degrees=(-20,+20)),
    #     transforms.ToTensor(),
    #     normalize,
    # ]))

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            pin_memory=True,
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        pin_memory=True,
    )

    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size, shuffle=True
    # )

    num_examples = {"trainset": len(train_dataset), "testset": len(val_dataset)}

    return train_loader, valid_loader, num_examples
