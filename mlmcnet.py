import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import argparse
from pathlib import Path

from dataset import UrbanSound8KDataset



torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum Value")
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


def main(args):
    train_loader = torch.utils.data.DataLoader(
          UrbanSound8KDataset(‘UrbanSound8K_train.pkl’, mode),
          batch_size=args.batch_size, shuffle=True,
          num_workers=args.worker_count, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
         UrbanSound8KDataset(‘UrbanSound8K_test.pkl’, mode),
         batch_size=args.batch_size, shuffle=False,
         num_workers=args.worker_count, pin_memory=True)

    # Cross entropy loss as declared
    criterion = nn.CrossEntropyLoss()
    # Using SGD with momentum as declared
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # Get mode from args, LMC, MC, or MLMC expected
    mode = args.mode
    if(mode == 'LMC'):
        model = CNN(height=85, width=41, channels=3, class_count=10, dropout=args.dropout)
    elif(mode == 'MC'):
        model = CNN(height=85, width=41, channels=3, class_count=10, dropout=args.dropout)
    # TODO Need to add support for MLMC combined since sizes are increased
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)


    model = model.to(DEVICE)

    # Run training for all epochs, validation as expected
    for epoch in range(0:args.epochs):
        print("Epoch : " + str(epoch+1))
        trainer(train_loader, model, criterion, args.val_frequency, DEVICE)

        if(epoch+1 % val_frequency == 0):
            validate(val_loader, model, criterion, DEVICE)


# Training function for an epoch
def trainer(train_loader, model, criterion, val_frequency, device):
    #Put model into training mode
    model.train()

    for i, (input, target, filename) in enumerate(train_loader):
        # Move stuff to cuda
        input = input.to(device)
        target = target.to(device)

        logits = model.forward(input)

        loss = criterion(logits,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Validation function
def validate(val_loader, model, criterion, device):
    results = {"preds": [], "labels": []}
    total_loss = 0

    # Put model into evaluation mode
    model.eval()

    with torch.no_grad():
        for i, (input, target, filename) in enumerate(val_loader):
            # Move stuff to cuda
            input = input.to(device)
            target = target.to(device)

            logits = model(input)
            loss = criterion(logits,target)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            results["preds"].extend(list(preds))
            results["labels"].extend(list(target.cpu().numpy()))

    accuracy = compute_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    average_loss = total_loss / val_loader.len()

    print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")






def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)
