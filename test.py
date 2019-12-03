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

import pandas as pd

from models.han_bilstm import hanLSTM
from models.lstm import lstm_cell 
from models.attention import Attention
from optims import Optim

import numpy as np
from tqdm import tqdm
torch.manual_seed(1234)
CUDA_LAUNCH_BLOCKING=1

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

batch_size = 2
print('loading data...')
start_time = time.time()
datas = torch.load('./data/final_data_3')  
print('loading time cost: %.3f' % (time.time()-start_time))
trainset, valset = datas['train'], datas['val']
# trainloader = get_loader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = get_loader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
print('dataloader prepared')

model = hanLSTM(doc_len=8, text_len=20, vocab_size=50002, embed_dim=300, word_hidden_size=300, 
                sent_hidden_size=300, title_vocab_size=50002, title_hidden_size=300, 
                linear_out_size_2=1200, linear_out_size_1=600, num_classes=300, dropout=0.5)
# model.cuda()
model.load_state_dict(torch.load('./trained_models/0.42536141672099454_checkpoint.pt', map_location=torch.device('cpu'))['model'])

class Dict(object):
    def __init__(self, data=None, lower=False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower
        # Special entries will not be pruned.
        self.special = [] 

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    # Load entries from a file.
    def loadFile(self, filename):
        for line in open(filename):
            fields = line.split()
            label = ' '.join(fields[:-1])
            idx = int(fields[-1])
            self.add(label, idx)

    # Write entries to a file.
    def writeFile(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def loadDict(self, idxToLabel):
        for i in range(len(idxToLabel)):
            label = idxToLabel[i]
            self.add(label, i)

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size):
        if size >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)
        newDict = Dict()
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i in idx[:size]:
            newDict.add(self.idxToLabel[i.item()])

        return newDict

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        vec = [x for x in flatten(vec)]

        return torch.LongTensor(vec)

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop):
        labels = []

        for i in idx:
            if i == stop:
                break
            labels += [self.getLabel(i)]

        return labels

class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def initVocabulary(name, dataFile, vocabFile, vocabSize, sep=' ', char=False):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = Dict()
        vocab.loadFile(vocabFile)  
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize, sep=sep, char=char)  
        vocab = genWordVocab

    return vocab

text_vocab = initVocabulary('text', None, './data/text_dict', 50000, ' ', False)
pvs_vocab = initVocabulary('pvs', None, './data/pvs_dict', 50000, ' ', False)

model.eval()
y_pred_top = []
i = 0
for x_list, y in valloader:
    bx = Variable(x_list[0].type(torch.LongTensor))
    bxt = Variable(x_list[1].type(torch.LongTensor))
    y_pre = model(bx, bxt)
    y_label_top = torch.topk(y_pre, 15, dim=1)[1].data
    y_pred_top.extend(y_label_top.tolist())
    i+=2
    if(i==100):
      break

res = []
for i in range(100):
  item = []
  abstract = " "
  for j in range(8):
    for k in range(20):
      if (text_vocab.getLabel(int(valset[i][0][0][j][k].item())) != '<blank>'):
        abstract+=text_vocab.getLabel(int(valset[i][0][0][j][k].item()))
        abstract+=" "
  item.append(abstract)
  title = " "
  for k in range(20):
    if (text_vocab.getLabel(int(valset[i][0][1][k].item())) != '<blank>'):
      title+=text_vocab.getLabel(int(valset[i][0][1][k].item()))
      title+=" "
  item.append(title)
  for k in range(300):
    if int(valset[i][1][k].item()) == 1:
      item.append(pvs_vocab.getLabel(k))
  top_15 = []
  for label in y_pred_top[i]:
    top_15.append(pvs_vocab.getLabel(label))
  item.extend(top_15)
  res.append(item)

cols = ['abstract', 'title', 'ground truth venue']
for i in range(1,16):
  cols.append('predicted_venue_'+str(i))
result = pd.DataFrame(res, columns = cols)

result.to_csv("result.csv")
