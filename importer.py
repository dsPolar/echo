import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

import argparse
from pathlib import Path

from dataset import UrbanSound8KDataset



torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-2, type=float, help="Learning rate")
parser.add_argument("--data-aug-hflip", action="store_true")
parser.add_argument("--data-aug-brightness", default=0, type=float)
parser.add_argument("--data-aug-rotation", default=0, type=float)
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--mode",
    default='LMC',
    type=string,
    help="LMC, MC, MLMC",
)





train_loader = torch.utils.data.DataLoader(
      UrbanSound8KDataset(‘UrbanSound8K_train.pkl’, mode),
      batch_size=args.batch_size, shuffle=True,
      num_workers=args.worker_count, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
     UrbanSound8KDataset(‘UrbanSound8K_test.pkl’, mode),
     batch_size=args.batch_size, shuffle=False,
     num_workers=args.worker_count, pin_memory=True)


class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.dropout = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv1)

        self.norm1 = nn.BatchNorm2d(
            num_features=32,
        )

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1),
        )
        self.initialise_layer(self.conv2)

        self.norm2 = nn.BatchNorm2d(
            num_features=64,
        )

        self.pool2 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))

        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1),
        )

        self.initialise_layer(self.conv3)

        self.norm3 = nn.BatchNorm2d(
            num_features = 64,
        )

        self.conv4 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1),
        )
        self.initialise_layer(self.conv4)

        self.norm4 = nn.BatchNorm2d(
            num_features = 64,
        )

        self.pool4 = nn.MaxPool2d(kernel_size= (2,2), stride = (2,2))
        # 15488 = 11x22x64
        # Shape after pool4
        self.hfc = nn.Linear(15488,1024)

        self.fc1 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc1)


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(images))
        x = self.norm1(x)
        x = F.relu(self.conv2(self.dropout(x)))
        x = self.norm2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.norm3(x)

        x = F.relu(self.conv4(self.dropout(x)))
        x = self.norm4(x)
        x = self.pool4(x)

        x = torch.flatten(x, start_dim=1)
        #ReLU or sigmoid here is up for debate since it is not included in paper
        #Going with sigmoid to match fc1
        x = F.sigmoid(self.hfc(x))
        x = F.sigmoid(self.fc1(x))
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)



for i, (input, target, filename) in enumerate(train_loader):
#           training code

for i, (input, target, filename) in enumerate(val_loader):
#           validation code