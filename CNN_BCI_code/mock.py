import os
import shutil
import time
import gc
#import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import utils as vutils
from sklearn.metrics import confusion_matrix

best_cls1 = 0
from torch.utils.model_zoo import load_url

class classifier(nn.Module):

    def __init__(self, num_classes=4):
        super(classifier, self).__init__()
        conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=0)
        pool1 = nn.MaxPool2d(2, stride=2)
        conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        dropout1 = nn.Dropout(p=0.5)
        flat1 = nn.Flatten()

        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool1,
            conv2,
            nn.ReLU(),
            dropout1,
            flat1
        )

        self.fc_module = nn.Sequential(
            #nn.Linear(100*16*16, 100, bias = True), #layer*2
            #nn.Linear(16*11*11, 100, bias=True),
            nn.Linear(64*23*157,64, bias = True), #layer*3
            nn.ReLU(),
            nn.Linear(64,4, bias=True)
        )

        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)
        #make linear
        dim = 1
        for d in out.size()[1:]:
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return F.softmax(out, dim=1)
        
def transfer_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            print(param)
            if name.split(".")[0] == 'conv_module':
              param.requires_grad = False
        transfer_freeze(child)

from torch.utils.model_zoo import load_url

#construct model on cuda if available
use_cuda = torch.cuda.is_available()

model = classifier()
if use_cuda:
  model = model.cuda()


import torchvision.datasets as datasets

def csv_loader(path):
    sample = torch.from_numpy(np.loadtxt(path, delimiter=",", dtype=np.float32))
    return sample
####Need to be defined####
dataset_path = "/home/ski1999/SWIN_data_fold"
model_weight_save_path = "/home/ski1999/CNN_BCI_code/ckp"
ckp_path = '/home/ski1999/CNN_BCI_code/ckp/checkpoint_mock_t60_2.pth.tar'
best_ckp_path = '/home/ski1999/CNN_BCI_code/ckp/Best_checkpoint_mock_t60_2.pth.tar'
tune_ckp_path = '/home/ski1999/CNN_BCI_code/ckp/Tune_checkpoint_mock_t60_2.pth.tar'
tune_best_ckp_path = '/home/ski1999/CNN_BCI_code/ckp/TuneBest_checkpoint_mock_t60_2.pth.tar'
epoch_n = 10
fine_tune = 'y' #'n'

#Data loading code
torch.cuda.empty_cache()
gc.collect()

"""
traindir = os.path.join(dataset_path, 'Train')
testdir = os.path.join(dataset_path, 'Test') #Change var name if want to change val set and test set
valdir = os.path.join(dataset_path, 'Val') # ""
"""
dir_1 = os.path.join(dataset_path, '1')
dir_2 = os.path.join(dataset_path, '2')
dir_3 = os.path.join(dataset_path, '3')
dir_4 = os.path.join(dataset_path, '4')
dir_5 = os.path.join(dataset_path, '5')
dir_6 = os.path.join(dataset_path, '6')
dir_7 = os.path.join(dataset_path, '7')
dir_8 = os.path.join(dataset_path, '8')
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                   std=[0.229, 0.224, 0.225])

traindir = dir_5
traindir_2 = dir_6
traindir_3 = dir_7
traindir_4 = dir_8
traindir_5 = dir_1
traindir_6 = dir_4
testdir = dir_2
valdir = dir_3

batch_size = 100
num_workers = 0
train_dataset = datasets.DatasetFolder(traindir,loader = csv_loader, extensions=('.csv'))
train_dataset_2 = datasets.DatasetFolder(traindir_2,loader = csv_loader, extensions=('.csv'))
train_dataset_3 = datasets.DatasetFolder(traindir_3,loader = csv_loader, extensions=('.csv'))
train_dataset_4 = datasets.DatasetFolder(traindir_4,loader = csv_loader, extensions=('.csv'))
train_dataset_5 = datasets.DatasetFolder(traindir_5,loader = csv_loader, extensions=('.csv'))
train_dataset_6 = datasets.DatasetFolder(traindir_6,loader = csv_loader, extensions=('.csv'))
test_dataset = datasets.DatasetFolder(testdir,loader = csv_loader, extensions=('.csv'))
val_dataset = datasets.DatasetFolder(valdir,loader = csv_loader, extensions=('.csv'))

trn_loader = DataLoader(
    train_dataset+train_dataset_2+train_dataset_3+train_dataset_4+train_dataset_5+train_dataset_6, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=False) #train_dataset

val_loader = DataLoader(val_dataset,
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True, drop_last=False)

test_loader = DataLoader(test_dataset,
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True, drop_last=False)

device =torch.device('cuda')

torch.cuda.empty_cache()

def save_checkpoint(state, filename=ckp_path):
    torch.save(state, filename)

def load_model(model, optimizer, file_path):
    if os.path.isfile(file_path):
        print("=> loading checkpoint '{}'".format(file_path))
        if torch.cuda.is_available():
          #Map model to be loaded to specified single gpu.
          loc = 'cuda:{}'.format((torch.cuda.current_device()))
          checkpoint = torch.load(resume_path_2, map_location=loc)
          print("use cuda")
        else:
          device = torch.device('cpu')
          checkpoint = torch.load(resume_path, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(file_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(file_path))
        
#loss
criterion = nn.CrossEntropyLoss()
#backpropagation method
learning_rate = 1e-6 #1e-4  #1e-3 #1e-6
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps = 1)
#optionally resume from a checkpoint
start_epoch = 0
resume_path = ckp_path
resume_path_2 = best_ckp_path

load_model(model, optimizer, resume_path_2)
if fine_tune == 'y':
  transfer_freeze(model)
  trn_loader = val_loader
  epoch_n = 1
  resume_path = tune_ckp_path
  resume_path_2 = tune_best_ckp_path
for param in model.parameters():
  print(param)
  
#hyper-parameters
num_epochs = 100
num_batches = len(trn_loader)

trn_loss_list = []
trn_acc_list = []
val_loss_list = []
val_acc_list = []
test_loss_list = []
test_acc_list = []
ens_acc_list = []

for epoch in range(num_epochs):
    model.train()

    trn_loss = 0.0
    start = time.time()

    total = 0
    correct = 0
    best_acc = 0

    for i, data in enumerate(trn_loader):
        x, label = data
        x = x.float()
        x = x.unsqueeze(1)
        if use_cuda:
            x = x.cuda()
            label = label.cuda()
        #grad init
        optimizer.zero_grad()
        #forward propagation
        model_output = model(x)
        _, predicted = torch.max(model_output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        #calculate loss
        loss = criterion(model_output, label)
        #back propagation
        loss.backward()
        #weight update
        optimizer.step()

        #trn_loss summary
        #print("loss item:", loss.item())
        trn_loss += loss.item()
        #print(trn_loss)
        #del (memory issue)
        del loss
        del model_output
        if (i + 1) % epoch_n == 0:
          print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | trn_acc: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, num_batches, trn_loss / 100, 100 * correct / total
                ))
          trn_loss_list.append(trn_loss / 100)
          trn_acc_list.append(100 * correct / total)
          trn_loss = 0.0
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, resume_path)
        if best_acc < (100 * correct / total):
          best_acc = 100 * correct / total
          save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
          }, resume_path_2)

    end = time.time()

    model.eval()

    correct = 0
    total = 0
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    classes = ('H_', 'L_')

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            #inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs, labels
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            e_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += e_loss.item()
            c = (predicted == labels).squeeze()

            for i in range(2):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                #print(labels[i])
                #print('Accuracy of %.2s : %.3f (%.2f) %%' % (
                #classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))

    print('(Test) Accuracy of the network on the test images: %.3f %%\n' % (
            100 * correct / total))
    test_loss_list.append(test_loss / len(test_loader))
    test_acc_list.append(100 * correct / total)
    print("train_loss_list = ", trn_loss_list)
    print("train_acc_list =", trn_acc_list)
    print("test_loss_list =", test_loss_list)
    print("test_acc_list =", test_acc_list)