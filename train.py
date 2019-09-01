import os
import codecs
import json as js
import argparse
import time
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

from sklearn import metrics

from models.han import hanLSTM
from models.lstm import lstm_cell 
from models.attention import Attention
from optims import Optim

import numpy as np
from tqdm import tqdm
torch.manual_seed(1234)


class dataset(torch.utils.data.Dataset):

    def __init__(self, text_data, label_data):
        self.text_data = text_data
        self.label_data = label_data
  
    def __getitem__(self, index):
        return [torch.from_numpy(x[index]).type(torch.FloatTensor) for x in self.text_data],\
               torch.from_numpy(self.label_data[index]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.label_data)
       

def get_loader(dataset, batch_size, shuffle, num_workers):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader

cwd=os.getcwd()
batch_size = 256
print('loading data...\n')
start_time = time.time()
datas = torch.load('./data/final_data')  
print('loading time cost: %.3f' % (time.time()-start_time))
trainset, valset = datas['train'], datas['val']
trainloader = get_loader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = get_loader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
print('dataloader prepared')

model = hanLSTM(10, 25, 400, 500, 50000, 400, 300, 300)
optim = Optim("adam", 0.01, 10, lr_decay=1e-5, start_decay_at=10)

optim.set_parameters(model.parameters())

param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

updates = 0
loss_function = nn.CrossEntropyLoss()
eval_interval = 100000

def eval_metrics(y_pred, y_true):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')

    return {'accuracy': accuracy, 'f1': f1,
            'precision': precision, 'recall': recall}

def train(epoch):
    model.train()
    global e, loss, updates, total_loss, start_time, report_total
    for e in range(1, epoch + 1):
        loop = tqdm(trainloader)
        for x_list, y in loop:
            bx = Variable(x_list[0].type(torch.LongTensor))
            by = Variable(y.type(torch.FloatTensor))
            model.zero_grad()
            y_pre = model(bx)
            loss = loss_function(y_pre, torch.max(by, 1)[1])
            loss.backward()
            for param in model.parameters():
                print(param.grad)
            optim.step()
            loop.set_description('Epoch {}/{}'.format(e, epoch))
            loop.set_postfix(loss=str(loss.item()))

def eval():
    model.eval()
    y_true, y_pred = [], []
    for x_list, y in valloader:
        bx, by = Variable(x_list[0].type(torch.LongTensor)), Variable(y)
        y_pre = model(bx)
        y_label = torch.max(y_pre, 1)[1].data
        y_true.extend(torch.max(y, 1)[1].tolist())
        y_pred.extend(y_label.tolist())

    score = {}
    result = eval_metrics(y_pred, y_true)
    print('Epoch: %d | Updates: %d | Train loss: %.4f | Accuracy: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f'
          % (e, updates, loss.data[0], result['accuracy'], result['f1'], result['precision'], result['recall']))
    score['accuracy'] = result['accuracy']
    score['f1'] = result['f1']

    return score

def main():
    train(30)

if __name__ == '__main__':
    main()