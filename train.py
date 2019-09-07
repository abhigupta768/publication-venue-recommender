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
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from sklearn import metrics

from models.han import hanLSTM
from models.lstm import lstm_cell 
from models.attention import Attention
from optims import Optim

import numpy as np
from tqdm import tqdm
torch.manual_seed(1234)

torch.cuda.set_device(0)
torch.cuda.manual_seed(1234)

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

batch_size = 128
print('loading data...')
start_time = time.time()
datas = torch.load('./data/final_data')  
print('loading time cost: %.3f' % (time.time()-start_time))
trainset, valset = datas['train'], datas['val']
trainloader = get_loader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = get_loader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
print('dataloader prepared')

model = hanLSTM(10, 25, 300, 300, 50000, 300, 300, 600)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=6e-4, weight_decay=3e-4)
# optimizer = optim.SGD(model.parameters(), lr=5e-3, weight_decay=1e-4, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True)
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

updates = 0
loss_function = nn.CrossEntropyLoss()

def eval_metrics(y_pred, y_true):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')

    return {'accuracy': accuracy, 'f1': f1,
            'precision': precision, 'recall': recall}

def save_model(path):
    model_state_dict = model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'optim': optimizer.state_dict()}
    torch.save(checkpoints, path)

print_interval = 200
max_val_acc = 0
def train(epoch):
    model.train()
    global e, loss, updates, total_loss, start_time, report_total, max_val_acc
    for e in range(1, epoch + 1):
        updates=0
        for x_list, y in trainloader:
            bx = Variable(x_list[0].type(torch.LongTensor)).cuda()
            by = Variable(y.type(torch.FloatTensor)).cuda()
            model.zero_grad()
            y_pre = model(bx)
            loss = loss_function(y_pre, torch.max(by, 1)[1])
            loss.backward()
            clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            updates+=1
            if updates % print_interval == 0:
              print('Epoch: {}, Update: {}, Training Loss: {}'.format(e, updates, loss.item()))
        print("\nEvaluating\n")
        score=eval()
        scheduler.step(score['accuracy'])
        if score['accuracy'] >= max_val_acc:
          print("Validation Accuracy increased to {} from {}. Saving Model...\n".format(score['accuracy'], max_val_acc))
          max_val_acc = score['accuracy']
          save_model('./trained_models/' + str(max_val_acc) + '_checkpoint.pt')
        model.train()

def eval():
    model.eval()
    y_true, y_pred = [], []
    for x_list, y in valloader:
        bx, by = Variable(x_list[0].type(torch.LongTensor)).cuda(), Variable(y).cuda()
        y_pre = model(bx)
        y_label = torch.max(y_pre, 1)[1].data
        y_true.extend(torch.max(y, 1)[1].tolist())
        y_pred.extend(y_label.tolist())

    score = {}
    result = eval_metrics(y_pred, y_true)
    print('Epoch: %d | Updates: %d | Train loss: %.4f | Accuracy: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f'
          % (e, updates, loss.item(), result['accuracy'], result['f1'], result['precision'], result['recall']))
    score['accuracy'] = result['accuracy']
    return score

def main():
    train(30)

if __name__ == '__main__':
    main()