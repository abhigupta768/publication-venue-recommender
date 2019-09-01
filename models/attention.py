import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_matmul(seq, weight, nonlinearity='tanh'):
    """
    Inputs:
        seq: tensor, [b, s, i]
        weight: tensor, [i, 1]
    Returns:
        tensor, [b, s]
    """

    seq = seq.transpose(0, 1)
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if nonlinearity == 'tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if s is None:
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze().transpose(0, 1)

def attention_mul(rnn_outputs, att_weights):
    """
    Inputs:
        rnn_outputs: tensor, [b, s, i]
        att_weights: tensor, [b, s]
    Return: [b, i]
    """

    rnn_outputs = rnn_outputs.transpose(0, 1)
    att_weights = att_weights.transpose(0, 1)
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0)

class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()

        self.input_size = input_size
        self.linear_project = nn.Linear(input_size, input_size)
        self.representation = nn.Parameter(torch.randn(input_size, 1))

    def forward(self, x):
        """
        Input: x: Output from the LSTM. x.size(): [b, s, i]
        Return: s: final representation weighted by attention weights. s.size(): [b, i]
        """

        batch_size, seq_len, input_size = x.size()
        # u.size(): [b, s, i]
        u = self.linear_project(x.contiguous().view(-1, input_size)).contiguous().view(batch_size, seq_len, -1)
        atten_weights = F.softmax(batch_matmul(u, self.representation), dim=1)
        s = attention_mul(x, atten_weights)
        return s