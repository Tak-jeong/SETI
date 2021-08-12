'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import timm
from sklearn.metrics import roc_auc_score

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path

from utils import progress_bar
from dataset import SETIDataset

from gridmask import GridMask
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (
    Compose, ShiftScaleRotate, Blur, Resize, Cutout, HorizontalFlip, RandomRotate90, VerticalFlip
)

parser = argparse.ArgumentParser(description='PyTorch SETI Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--img', default=224, type=int, help='image size')
parser.add_argument('--epoch',default=100, type=int, help='num of epoch')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_score = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = albumentations.Compose([
    Resize(args.img,args.img),
    ShiftScaleRotate(rotate_limit=15),
    albumentations.OneOf([
        GridMask(num_grid=3,rotate=15),
        GridMask(num_grid=(3,7)),
        GridMask(num_grid=3,mode=2)],p=1),
    ToTensorV2()
])

transform_test = albumentations.Compose([
    Resize(args.img,args.img),
    ToTensorV2()
])

path = Path('/home/datasets/SETI/')
csv_file = path.joinpath('train_labels.csv')

df = pd.read_csv(csv_file)
skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)


for train_idx, test_idx in skf.split(df, df.target):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    break

trainset=SETIDataset(df=train_df,path=path,transform=transform_train)
testset=SETIDataset(df=test_df,path=path,transform=transform_test)



trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=4)


# Model
print('==> Building model..')

net=timm.create_model('efficientnet_b4',pretrained=True,in_chans=1,num_classes=1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr,
                      weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20,30,40],gamma=0.5)

current_time=datetime.now().strftime('%b%d_%H-%M-%S')
log_dir=os.path.join(f'runs/{current_time}')
writer=SummaryWriter(log_dir)

# Training
def train():
    net.train()
    train_loss = 0
    all_targets = []
    all_predictions = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        all_targets.extend(targets.cpu().detach().numpy().tolist())
        all_predictions.extend(outputs.sigmoid().cpu().detach().numpy().tolist())
        roc_auc = roc_auc_score(all_targets, all_predictions)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | score: %.3f '
                     % (train_loss/(batch_idx+1), roc_auc))
        
    return train_loss/(batch_idx+1), roc_auc


def test():
    global best_score
    net.eval()
    test_loss = 0
    all_targets=[]
    all_predictions=[]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            all_targets.extend(targets.cpu().detach().numpy().tolist())
            all_predictions.extend(outputs.sigmoid().cpu().detach().numpy().tolist())
            roc_auc = roc_auc_score(all_targets, all_predictions)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | score: %.3f '
                         % (test_loss/(batch_idx+1), roc_auc))
        
    # Save checkpoint.
    if roc_auc > best_score:
        print('Saving..')
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/best_ckpt.pth')
        best_score = roc_auc
    
    return test_loss/(batch_idx+1), roc_auc


for epoch in range(start_epoch, start_epoch+args.epoch):
    print('\n[Epoch: %d]'%epoch)
    train_loss,train_roc_auc=train()
    test_loss, test_roc_auc=test()
    scheduler.step()

    writer.add_scalar('Loss/train',train_loss, epoch)
    writer.add_scalar('Loss/test',test_loss, epoch)
    writer.add_scalar('ROC_AUC/train',train_roc_auc, epoch)
    writer.add_scalar('ROC_AUC/test',test_roc_auc, epoch)


print('Saving..')
torch.save(net.state_dict(), './checkpoint/last_ckpt.pth')
writer.close()