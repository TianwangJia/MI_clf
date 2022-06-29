import argparse
import os
import numpy as np
from datetime import datetime
from scipy import rand
from sklearn import datasets
from tqdm import tqdm
import random
import torch.nn as nn

from EEGNet import *
from ..datasets import *

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',
                    type=int,
                    default=1000,
                    help='number of epochs of training')
parser.add_argument('--train_data_path',
                    type=str,
                    default='./data/train/',
                    help='path to the datasets')
parser.add_argument('--log-path',
                    type=str,
                    default='./log/',
                    help='path to store the logs')
parser.add_argument('--output_path',
                    type=str,
                    default='./output/',
                    help='path to save model and outputs')
parser.add_argument('--batch_size',
                    type=int,
                    default=64,
                    help='size of the batches')
parser.add_argument('--lr',
                    type=float,
                    default=0.0002,
                    help='adam: learning rate')
parser.add_argument('--b1',
                    type=float,
                    default=0.5,
                    help='adam: decay of the first order momentum of gradient')
parser.add_argument(
    '--b2',
    type=float,
    default=0.999,
    help='adam: decay of the second order momentum of gradient')
parser.add_argument('--ea', type=bool, default=True, help='whether to ea data')
parser.add_argument('--csp', type=bool, default=True, help='whether transform data into csp space')
parser.add_argument('--car', type=bool, default=False, help='whether to common average referencing data')
parser.add_argument('--val_user', type=int, default=0, help='val user')

args = parser.parse_args()
print(args)

# Create output and log directories
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('%ssave_models/%s' % (args.output_path, timestamp))
os.makedirs('%sruns/%s' % (args.log_path, timestamp))

# Get data
seed = random.randint(1, 100)
train_dataloader = DataLoader(MIDataset(random_state=seed, is_car = args.car, is_ea=args.ea,is_csp=args.csp, start=0, end=750, val_user=args.val_user),
                            batch_size=args.batch_size,
                            shuffle=True)
test_dataloader = DataLoader(MIDataset(random_state=seed, mode='test',is_car=args.car, is_csp=args.csp,
                                    is_ea=args.ea, start=0, end=750, val_user=args.val_user),
                            batch_size=args.batch_size,
                            shuffle=True)

# Device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EEGNet(time=750, drop_out=0.5).to(device)

# Losses
criterion = nn.CrossEntropyLoss()

# Optim
optimizer = torch.optim.Adam(model.parameters(),
                            lr=args.lr,
                            betas=(args.b1, args.b2))

# Initialize weights
model.apply(weights_init)

writer = SummaryWriter('%sruns/%s' % (args.log_path, timestamp))
best_test_acc = 0
best_epoch = 0
for epoch in tqdm(range(args.n_epochs), desc='epoch', position=1):

    # Training
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        if epoch == 0 and batch_idx == 0:
            writer.add_graph(model, X)
            
        # print('len(train):', len(train_dataloader))
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y, y_hat)
        loss.backward()
        optimizer.step()

        train_loss += loss
        pred = y_hat.max(-1, keepdim=True)[1]
        y_true = y.max(-1, keepdim=True)[1]
        train_acc += pred.eq(y_true).sum().item()

    train_loss = train_loss / len(train_dataloader.dataset)
    train_acc = train_acc / len(train_dataloader.dataset)  # 运行每个epoch的loss和acc

    # Testing
    model.eval()
    test_loss = 0
    test_acc = 0
    # test_recall = 0

    for batch_idx, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            y_hat = model(X)

        loss = criterion(y_hat, y)
        test_loss += loss
        pred = y_hat.max(-1, keepdim=True)[1]
        y_true = y.max(-1, keepdim=True)[1]
        test_acc += pred.eq(y_true).sum().item()

    test_loss = test_loss / len(test_dataloader.dataset)
    test_acc = test_acc / len(test_dataloader.dataset)

    if best_test_acc <= test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
        torch.save(
            model.state_dict(),
            "%ssave_models/%s/bestmodel.pth" % (args.output_path, timestamp))

    writer.add_scalars('loss', {
        'train_loss': train_loss,
        'test_loss': test_loss
    },
                    global_step=epoch)
    writer.add_scalars('acc', {
        'train_acc': 100 * train_acc,
        'test_acc': 100 * test_acc
    },
                    global_step=epoch)

writer.close()

print('\nBest test_acc:{:.4f} %'.format(100 * best_test_acc))
print('Best epoch: ', best_epoch)
print('val user: ',args.val_user )
