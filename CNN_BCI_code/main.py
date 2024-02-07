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

class classifier_org(nn.Module):

    def __init__(self, num_classes=4):
        super(classifier_org, self).__init__()
        conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=0)
        conv2 = nn.Conv2d(4, 16, kernel_size=5, padding=0)
        pool1 = nn.MaxPool2d(2, stride=2)
        conv3 = nn.Conv2d(16, 64, kernel_size=5, padding=0)
        conv4 = nn.Conv2d(64, 256, kernel_size=5, padding=0)
        pool2 = nn.MaxPool2d(2, stride=2)
        bn1 = nn.BatchNorm1d(256)
        # bn2 = nn.BatchNorm1d(4)

        self.conv_module = nn.Sequential(
            conv1,
            conv2,
            nn.ReLU(),
            pool1,
            conv3,
            conv4,
            nn.ReLU(),
            pool2
        )

        self.fc_module = nn.Sequential(
            nn.Linear(256*7*7, 256, bias=True),
            bn1,
            nn.ReLU(),
            nn.Linear(256, 4, bias=True)
        )

        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)
        # make linear
        dim = 1
        for d in out.size()[1:]:
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return F.softmax(out, dim=1)


class classifier(nn.Module):

    def __init__(self, num_classes=4):
        super(classifier, self).__init__()

        ########################################################################
        # Construct the layers of model according to the instruction
        #
        # self.features : an instance of nn.Sequence which can be acquired from the arguments
        # self.conv6    : 2D Convolutional Layer with out_channel=1024, kernel_size=3, padding=1
        # self.relu     : ReLU Layer
        # self.fc       : Fully Connected Layer which output_size is num_classes
        #
        # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py 참고
        ########################################################################
        conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(4, 16, kernel_size=3, padding=0)
        pool1 = nn.MaxPool2d(2, stride=2)
        conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=0)
        conv4 = nn.Conv2d(64, 256, kernel_size=3, padding=0)
        pool2 = nn.MaxPool2d(2, stride=2)
        conv5 = nn.Conv2d(256, 1024, kernel_size=3, padding=0)
        conv6 = nn.Conv2d(1024, 4096, kernel_size=3, padding=0)
        pool3 = nn.MaxPool2d(2, stride=2)
        bn1 = nn.BatchNorm1d(100)
        # bn2 = nn.BatchNorm1d(4)

        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            pool1,
            #conv2,
            conv3,
            nn.ReLU(),
            conv4,
            nn.ReLU(),
            pool2,
            conv5,
            nn.ReLU(),
            conv6,
            nn.ReLU(),
            pool3
        )

        self.fc_module = nn.Sequential(
            #nn.Linear(100*16*16, 100, bias = True), #layer*2
            #nn.Linear(16*11*11, 100, bias=True),
            nn.Linear(144*16*16,100, bias = True), #layer*3
            bn1,
            nn.ReLU(),
            nn.Linear(100,2, bias=True)
        )

        # gpu로 할당
        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):

        ########################################################################
        # Construct the forward pass of model
        # The order of forward pass is originally same as layer declaration
        #
        # You should assign the results of appropriate layers to self.feature_map and self.pred
        ########################################################################
        out = self.conv_module(x)
        # make linear
        dim = 1
        for d in out.size()[1:]:
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return F.softmax(out, dim=1)
#   def _classifier(pretrained):
from torch.utils.model_zoo import load_url

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

model = classifier()
if use_cuda:
  model = model.cuda()


import torchvision.datasets as datasets

def csv_loader(path):
    sample = torch.from_numpy(np.loadtxt(path, delimiter=",", dtype=np.float32))
    return sample
#### Need to be defined ####
dataset_path = "/home/ski1999/CNN_plv_hamv1_fold/fold_4" #checkpoint path 아니라 dataset path!!!
model_weight_save_path = "/home/ski1999/CNN_BCI_code/ckp"
ckp_path = '/home/ski1999/CNN_BCI_code/ckp/checkpoint_hamv1_PLV_t60_4.pth.tar'
best_ckp_path = '/home/ski1999/CNN_BCI_code/ckp/Best_checkpoint_hamv1_PLV_t60_4.pth.tar'
ensemble_result_path = '/home/ski1999/CNN_BCI_code/Ensemble/plvhamv1_Ensemble_150_Result_4.csv'
aug_size = 150
epoch_n = 100

# Data loading code
torch.cuda.empty_cache()
gc.collect()

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

traindir = dir_1
traindir_2 = dir_8
traindir_3 = dir_2
traindir_4 = dir_3
traindir_5 = dir_6
traindir_6 = dir_7
testdir = dir_4
valdir = dir_5
"""
batch_size = 100
num_workers = 0
train_dataset = datasets.DatasetFolder(traindir,loader = csv_loader, extensions=('.csv'))
"""
train_dataset_2 = datasets.DatasetFolder(traindir_2,loader = csv_loader, extensions=('.csv'))
train_dataset_3 = datasets.DatasetFolder(traindir_3,loader = csv_loader, extensions=('.csv'))
train_dataset_4 = datasets.DatasetFolder(traindir_4,loader = csv_loader, extensions=('.csv'))
train_dataset_5 = datasets.DatasetFolder(traindir_5,loader = csv_loader, extensions=('.csv'))
train_dataset_6 = datasets.DatasetFolder(traindir_6,loader = csv_loader, extensions=('.csv'))
"""
test_dataset = datasets.DatasetFolder(testdir,loader = csv_loader, extensions=('.csv'))
val_dataset = datasets.DatasetFolder(valdir,loader = csv_loader, extensions=('.csv'))

trn_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=False) #train_dataset+train_dataset_2+train_dataset_3+train_dataset_4+train_dataset_5+train_dataset_6

val_loader = DataLoader(val_dataset,
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True, drop_last=False)

test_loader = DataLoader(test_dataset,
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True, drop_last=False)

device =torch.device('cuda')

torch.cuda.empty_cache()

# samples 순서대로 나열되어있는 파일 만들기
# 같은 사람 - 같은 시간대에 있는 데이터들 순서대로 인덱싱 파일 만들기

allFiles, _ = map(list, zip(*test_loader.dataset.samples))
indexFiles = []
len_file = 0

for i, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    for j in range(inputs.size()[0]):
        aug_label = allFiles[i * batch_size + j]
        aug_label = aug_label.split("/")[7] #8fold 한번에 들어있는 파일은 8, 아닌 건 7
        aug_label = aug_label.split("_")
        #print(aug_label)
        aug_time = aug_label[5].split(".")
        indexFiles.append([aug_label[4],int(aug_time[0]),int(aug_label[1])])
        len_file = len_file+1
len_file = int(len_file/aug_size)
print(len_file)
#print(indexFiles)

def save_checkpoint(state, filename=ckp_path):
    torch.save(state, filename)

def load_model(model, optimizer, args, file_path):
    if os.path.isfile(file_path):
        print("=> loading checkpoint '{}'".format(file_path))
        if args.gpu is None:
            checkpoint = torch.load(file_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(0)
            checkpoint = torch.load(file_path, map_location=loc)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# loss
criterion = nn.CrossEntropyLoss()
# backpropagation method
learning_rate = 1e-5 #1e-6  #1e-3 #1e-4
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps = 1)
# optionally resume from a checkpoint
start_epoch = 0
resume_path = ckp_path
resume_path_2 = best_ckp_path

if os.path.isfile(resume_path):
    print("=> loading checkpoint '{}'".format(resume_path))
    if torch.cuda.is_available():
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format((torch.cuda.current_device()))
        checkpoint = torch.load(resume_path_2, map_location=loc)
        print("use cuda")
    else:
        device = torch.device('cpu')
        checkpoint = torch.load(resume_path, map_location=device)

    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # loss = checkpoint['loss']
    print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(resume_path))

# hyper-parameters
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
    val_total = 0
    val_correct = 0

    for i, data in enumerate(trn_loader):
        x, label = data
        x = x.float()
        x = x.unsqueeze(1)
        if use_cuda:
            x = x.cuda()
            label = label.cuda()
        # grad init
        optimizer.zero_grad()
        # forward propagation
        model_output = model(x)
        _, predicted = torch.max(model_output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        # calculate loss
        loss = criterion(model_output, label)
        # back propagation
        loss.backward()
        # weight update
        optimizer.step()

        # trn_loss summary
        #print("loss item:", loss.item())
        trn_loss += loss.item()
        #print(trn_loss)
        # del (memory issue)
        del loss
        del model_output

        # 학습과정 출력
        if (i + 1) % epoch_n == 0:  # every 100 mini-batches (Validation), 1 = no aug, 50 = aug*30
            with torch.no_grad():  # very very very very important!!!
                val_loss = 0.0
                for j, val in enumerate(val_loader):
                    val_x, val_label = val
                    val_x = val_x.float()
                    val_x = val_x.unsqueeze(1)
                    if use_cuda:
                        val_x = val_x.cuda()
                        val_label = val_label.cuda()
                    val_output = model(val_x)
                    _, predicted = torch.max(val_output, 1)
                    val_total += val_label.size(0)
                    val_correct += (predicted == val_label).sum().item()
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss

            print(
                "epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | trn_acc: {:.4f}| val_acc: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, num_batches, trn_loss / 100, val_loss / len(val_loader),
                    100 * correct / total, 100 * val_correct / val_total
                ))
            trn_loss_list.append(trn_loss / 100)
            val_loss_list.append(val_loss / len(val_loader))
            trn_acc_list.append(100 * correct / total)
            val_acc_list.append(100 * val_correct / val_total)
            trn_loss = 0.0
            new_cls1 = val_acc_list[-1]
            #print("best_acc = {}".format(new_cls1))
            if best_cls1 < new_cls1:
                best_cls1 = new_cls1
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 }, resume_path_2)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, resume_path)

    end = time.time()

    model.eval()

    correct = 0
    total = 0
    test_loss = 0.0
    #class_correct = list(0. for i in range(4))
    #class_total = list(0. for i in range(4))
    #classes = ('H', 'HL', 'LH', 'LL')
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

            for j in range(inputs.size()[0]):
                if indexFiles[i * batch_size + j][0][0] == classes[predicted[j]][0]:
                    indexFiles[i*batch_size+j].append(1)
                else:
                    indexFiles[i * batch_size + j].append(0)

            for i in range(2):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                #print(labels[i])
                #print('Accuracy of %.2s : %.3f (%.2f) %%' % (
                #classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))

    print('(Test) Accuracy of the network on the test images: %.3f %%\n' % (
            100 * correct / total))

    eval_indexFiles = sorted(indexFiles)
    df_indexFiles = pd.DataFrame(indexFiles)
    df_indexFiles.to_csv(ensemble_result_path, index=False, mode='w', header=False)
    #print(eval_indexFiles)
    ensemble_label_predicted = []
    for t_a in range(len_file):
        ens_cnt = 0
        for t_b in range(aug_size): #should change if aug size change
            ens_cnt = ens_cnt + eval_indexFiles[(aug_size*t_a)+t_b][epoch+3]
        ens_cnt = round(ens_cnt/aug_size)
        ensemble_label_predicted.append(ens_cnt)
    ens_acc_list.append(sum(ensemble_label_predicted)*100/len_file)
    print('Ensemble Acc : %.3f %%\n' %(sum(ensemble_label_predicted)*100/len_file))

    test_loss_list.append(test_loss / len(test_loader))
    test_acc_list.append(100 * correct / total)
    print("train_loss_list = ", trn_loss_list)
    print("train_acc_list =", trn_acc_list)
    print("val_loss_list = ", val_loss_list)
    print("val_acc_list =", val_acc_list)
    print("test_loss_list =", test_loss_list)
    print("test_acc_list =", test_acc_list)
    print("ens_acc_list=", ens_acc_list)

print('Finished Training')
