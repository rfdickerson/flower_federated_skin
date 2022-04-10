import os
from typing import Tuple, Dict

import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def load_data(dataset_dir, batch_size=32) -> Tuple[
    torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict
]:

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
        transforms.RandomRotation(degrees=(-20,+20)),
        transforms.ToTensor(),
        normalize,
    ]))

    val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transforms.Compose([
        transforms.RandomRotation(degrees=(-20,+20)),
        transforms.ToTensor(),
        normalize,
    ]))

    # test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transforms.Compose([
    #     transforms.RandomRotation(degrees=(-20,+20)),
    #     transforms.ToTensor(),
    #     normalize,
    # ]))

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=True
    )

    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size, shuffle=True
    # )

    num_examples = {"trainset": len(train_dataset), "testset": len(val_dataset)}

    return train_loader, valid_loader, num_examples
