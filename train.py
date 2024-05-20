import logging
import string
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torcheval.metrics.functional import multiclass_f1_score
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split

from model import ConvNet
from dataset import AngleDataset


# Set up basic logging configuration with a specific format and level.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%a %d %b %Y %H:%M:%S"
)

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, data_loader, criterion, optimizer, epoch):
    model.train()
    LOG_INTERVAL = 100
    total_acc, total_count, total_loss = 0, 0, 0


    for idx, (data, label) in enumerate(data_loader):
        # Optimize the parameters of the neuro network through backpropagation process.
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Calculate the variable used for logging.
        predicted_label = output.data.max(dim=1, keepdim=True)[1]
        total_acc += predicted_label.eq(label.data.view_as(predicted_label)).cpu().sum()
        total_count += data.shape[0]
        total_loss += loss.item()

        # Log the training loss every logging interval.
        if idx > 0 and idx % LOG_INTERVAL == 0:
            logging.info(f'| epoch {epoch} | {idx:2d}/{len(data_loader)} batches '
                         f'| accuracy {total_acc / total_count:.4f}'
                         f'| loss {loss.item():.4f}')
            total_acc, total_count, total_loss = 0, 0, 0


def eval(model, data_loader):
    model.eval()
    num_correct, total_loss = 0, 0
    predicted_labels, true_labels = [], []

    with torch.no_grad():
        for idx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            total_loss += loss
            predicted_label = output.data.max(dim=1, keepdim=True)[1]
            num_correct += predicted_label.eq(label.data.view_as(predicted_label)).cpu().sum()
            current_batch_size = predicted_label[0]
            for i in range(current_batch_size):
                predicted_labels.append(predicted_label[i][0].item())
                true_labels.append(label[i].item())

    accuracy = 100. * num_correct / len(data_loader.dataset)
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    logging.info(f'Accuracy on test dataset: {accuracy}%')
    logging.info(f'Macro-average F1 Score: {f1_macro}')
    eval_loss = loss / len(data_loader)

    return eval_loss


class EarlyStopping:
    """EarlyStopping class to monitor a given validation loss and stop the training process
    when the performance has not improved over a specified number of consecutive epochs.

    This technique is used to prevent overfitting and save computational resources by
    ending the training early once the model's improvement has plateaued.
    """

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def parse_arguments():
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train the model.")
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Spcifies learing rate for optimizer. (default: 1e-3)')
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs. (default: 50)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size for data loaders. (default: 64)'
    )
    parser.add_argument(
        '--class_num', type=int, default=26,
        help='Number of the categories. (default: 26)'
    )
    parser.add_argument(
        '--is_static', type=bool, default = True,
        help='Specifies whether the datasets are statically (True) or dynamically (False) collected. (dafault: True)'
    )
    parser.add_argument(
        '--dataset', action='append', type=int,
        help='Specifies the datasets selected in the experiment. (action:append)'
    )
    return parser.parse_args()

def generate_static_dataset():
    alphabet_list = list(string.ascii_lowercase)
    paths = []
    for alpha in alphabet_list:
        path = 'bfa/bfa_' + alpha + '.npy'
        paths.append(path)
    datasets = []
    for idx, path in enumerate(paths):
        datasets.append(AngleDataset(path, idx, type='static'))
    return ConcatDataset(datasets)

if __name__ == '__main__':
    opt = parse_arguments()
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2

    # Generate a dataset, randomly partition it into training, validation, and test sets
    # according to predefined proportions, for subsequent machine learning model training and evaluation.
    combined_dataset = generate_static_dataset()
    train_size = int(TRAIN_RATIO * len(combined_dataset))
    val_size = int(VAL_RATIO * len(combined_dataset))
    test_size = len(combined_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [train_size, val_size, test_size])

    criterion = torch.nn.CrossEntropyLoss()
    model = ConvNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    train_dataloader = DataLoader(train_dataset, opt.batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, opt.batch_size, shuffle=True)
    early_stop = EarlyStopping()

    for epoch in range(1, opt.epochs + 1):
        train(model, train_dataloader, criterion, optimizer, epoch)
        eval_loss = eval(model, valid_dataloader)
        if early_stop(eval_loss):
            break

    loss = eval(model, test_dataloader)
    logging.info('Training Finished')
    logging.info(f'在测试集上的loss: {loss}')
