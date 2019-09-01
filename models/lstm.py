import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class lstm_cell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2, dropout = 0, bi = False, batch_first = True):
        super(lstm_cell, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bi = bi
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            batch_first = self.batch_first,
                            dropout = self.dropout,
                            bidirectional = self.bi)
    
    def forward(self, x):
        """
        Input: x: Tensor, output from embedding layer. x.size(): [b, s, i]
        Return: output. output.size(): [b, s, h]
        """
        output, (_, _) = self.lstm(x, None)
        return output