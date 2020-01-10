#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import UrbanSound8KDataset


import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train LMC, MC or MLMC",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
#default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
#parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum Value")
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
#Dima results are at 50 epochs
parser.add_argument(
    "--epochs",
    default=50,
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
    help="LMC, MC, MLMC",
)
parser.add_argument("--checkpoint-path", default="checkpoints", type=Path)
parser.add_argument("--checkpoint-frequency", type=int, default=1, help="Save a checkpoint every N epochs")
parser.add_argument("--resume", default=0, type=int)
parser.add_argument("--resume-checkpoint", type=Path)




class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    mode = args.mode
    if args.resume == 1:
        if args.resume_checkpoint.exists():
            state_dict = torch.load(args.resume_checkpoint)
            print(f"Loading model from {args.resume_checkpoint}")
            model.load_state_dict(state_dict)

    train_loader = torch.utils.data.DataLoader(
          UrbanSound8KDataset("UrbanSound8K_train.pkl", mode),
          batch_size=args.batch_size, shuffle=True,
          num_workers=args.worker_count, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
         UrbanSound8KDataset("UrbanSound8K_test.pkl", mode),
         batch_size=args.batch_size, shuffle=False,
         num_workers=args.worker_count, pin_memory=True)

    criterion = nn.CrossEntropyLoss()


    if(mode == 'LMC'):
        model = CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout, mode=1)
    elif(mode == 'MC'):
        model = CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout, mode=2)
    elif(mode == 'MLMC'):
        model = CNN(height=145, width=41, channels=1, class_count=10, dropout=args.dropout, mode=3)


    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)



    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE, args
    )
    print("EPOCHS")
    print(args.epochs)

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()


class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float, mode: int):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.mode = mode

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
            out_channels = 32,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1),
        )
        self.initialise_layer(self.conv2)

        self.norm2 = nn.BatchNorm2d(
            num_features=32,
        )

        self.pool2 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))

        self.conv3 = nn.Conv2d(
            in_channels = 32,
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
        # Shape after pool4 if round up

        # 13440 = 10*21*64
        # Shape after pool4 if round down

        # 23040 = 10*36*64
        # Shape after pool4 if MLMC Combined set
        if((mode == 1) or (mode == 2)):
            linear = 13440
        else:
            linear = 23040
        self.hfc = nn.Linear(linear,1024)
        self.initialise_layer(self.hfc)

        self.fc1 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(images))
        # 85x41x32 here

        x = self.norm1(x)
        x = F.relu(self.conv2(self.dropout(x)))
        x = self.norm2(x)
        x = self.pool2(x)
        # 42x20x64 here

        x = F.relu(self.conv3(x))
        x = self.norm3(x)

        x = F.relu(self.conv4(self.dropout(x)))
        x = self.norm4(x)
        x = self.pool4(x)
        # 21x10x64 here

        x = torch.flatten(x, start_dim=1)
        #ReLU or sigmoid here is up for debate since it is not included in paper
        #Going with sigmoid to match fc1
        x = torch.sigmoid(self.hfc(x))
        x = torch.sigmoid(self.fc1(x))
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        args,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.args = args

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for i, (batch, labels, filename) in enumerate(self.train_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()


                ## TASK 1: Compute the forward pass of the model, print the output shape
                ##         and quit the program
                logits = self.model.forward(batch)


                ## TASK 7: Rename `output` to `logits`, remove the output shape printing
                ##         and get rid of the `import sys; sys.exit(1)`

                ## TASK 9: Compute the loss using self.criterion and
                ##         store it in a variable called `loss`
                loss = self.criterion(logits, labels)

                ## TASK 10: Compute the backward pass
                loss.backward()

                ## TASK 12: Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.validate()
                self.model.train()
            if (epoch + 1) % self.args.checkpoint_frequency or (epoch + 1) == epochs:
                print(f"Saving model to {self.args.checkpoint_path}")
                torch.save(self.model.state_dict(), self.args.checkpoint_path)



    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, (batch, labels, filename) in enumerate(self.val_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        perclass = compute_perclass_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )

        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )

        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print("AC Unit", perclass[0])
        print("Car Horn", perclass[1])
        print("Children", perclass[2])
        print("Dog Bark", perclass[3])
        print("Drilling", perclass[4])
        print("Engine Idle", perclass[5])
        print("Gunshot", perclass[6])
        print("Jackhammer", perclass[7])
        print("Siren", perclass[8])
        print("Street Music", perclass[9])
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        torch.save({
            'args': self.args,
            'model': self.model.state_dict(),
            'accuracy': accuracy
        }, self.args.checkpoint_path)



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

def compute_perclass_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]):
    accuracies = []
    for i in range(0,10):
        accuracies.append(float((((labels == i) & (preds == i)).sum()) / (labels == i).sum()))
    return accuracies

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
      f"CNN_bn_"
      f"dropout={args.dropout}_"
      f"bs={args.batch_size}_"
      f"lr={args.learning_rate}_"
      f"momentum=0.9_"
      f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())
