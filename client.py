import warnings
from collections import OrderedDict
import os
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from torch import nn

import efficientnet
from dataset import load_data
from efficientnet import build_model, train, test

USE_FEDBN: bool = True
DATASET_DIR = os.path.join(Path.home(), "Dropbox/machine-learning/dataset")
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Flower client
class SkinLesionClient(fl.client.NumPyClient):

    def __init__(self,
                model: nn.Module,
                trainloader: torch.utils.data.DataLoader,
                testloader: torch.utils.data.DataLoader,
                num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.model.train()

        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
            self, parameters: List[np.ndarray], config: Dict[str, str]):

        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": accuracy}



def main() -> None:

    trainloader, testloader, num_examples = load_data(DATASET_DIR)

    model = efficientnet.build_model().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    client = SkinLesionClient(model, trainloader, testloader, num_examples)

    fl.client.start_numpy_client("[::]:8080", client)


if __name__ == "__main__":
    main()
