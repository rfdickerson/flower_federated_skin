import argparse

import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=10)

