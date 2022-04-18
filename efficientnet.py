from typing import Tuple, Dict
import os

import torch
import torchvision
from torch import nn
from tqdm import tqdm
from pathlib import Path

from dataset import load_data

DATASET_DIR = os.path.join(Path.home(), "Dropbox/machine-learning/dataset")
BATCH_SIZE = 32
LR = 0.001
MOMENTUM = 0.9
EPOCHS = 1


def build_model() -> nn.Module:
    """create the CNN based on efficientnet"""
    model_conv = torchvision.models.efficientnet_b0(pretrained=True)

    # don't fit the convolutional network parameters
    for param in model_conv.parameters():
        param.requires_grad = False

    # create a new classification layer
    fc = nn.Sequential(
        nn.Linear(in_features=1280, out_features=625),
        nn.BatchNorm1d(625),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=7),
    )

    model_conv.classifier = fc
    return model_conv


def train(
        net: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        epochs,
        device: torch.device,
) -> None:
    """Train the model on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

    print(f"Training {epochs} epochs(s) w/ {len(trainloader)} batches each")
    net.to(device)
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()



def test(
    net: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the network on the entire test set"""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    net.to(device)
    net.eval()

    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader, _ = load_data(DATASET_DIR)
    net = build_model().to(DEVICE)
    net.eval()
    print("Start training")
    train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
