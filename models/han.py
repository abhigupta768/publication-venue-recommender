import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lstm import lstm_cell
from models.attention import Attention

class hanLSTM(nn.Module):
    def __init__(self, doc_len, text_len, word_hidden_size, embed_dim,
                 vocab_size, sent_hidden_size, num_classes, linear_out_size_1, linear_out_size_2):
        super(hanLSTM, self).__init__()

        self.doc_len = doc_len
        self.text_len = text_len
        self.word_hidden_size = word_hidden_size
        self.embed_size = embed_dim
        self.sent_hidden_size = sent_hidden_size
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.linear_out_size_1 = linear_out_size_1
        self.linear_out_size_2 = linear_out_size_2
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.sent_wise_lstms = nn.ModuleList()
        self.sent_wise_attlstms = nn.ModuleList()
        for i in range(self.doc_len):
            self.sent_wise_lstms.append(nn.Sequential(lstm_cell(self.embed_size, self.word_hidden_size), nn.Dropout(p=0.2)))
            self.sent_wise_attlstms.append(Attention(self.word_hidden_size))
        self.doc_lstm = nn.Sequential(lstm_cell(self.word_hidden_size, self.sent_hidden_size),nn.Dropout(p=0.2))
        self.doc_attention = Attention(self.sent_hidden_size)
        self.linear_stack = nn.Sequential(nn.Linear(self.sent_hidden_size, self.linear_out_size_2), nn.ReLU(), 
                                          nn.Dropout(p=0.3), nn.Linear(self.linear_out_size_2, self.linear_out_size_1), nn.ReLU(), 
                                          nn.Linear(self.linear_out_size_1, self.num_classes))
    
    def forward(self, x):
        """
        Input:
            x: training data. x.size(): [b, d, s]
        Return:
            y: prediction. y.size(): [b, num_classes]
        Immediate:
            x_embed.size(): [b, d, s, embed_size]
            sent_outs[i].size(): [b, s, word_hidden_size]
            sent_att_outs[i].size(): [b, word_hidden_size, 1]
            doc_representation.size(): [b, d, word_hidden_size]
            doc_lstm.size(): [b, d, sent_hidden_size]
            att_doc_lstm.size(): [b, sent_hidden_size]
        """

        x_embed = self.embedding(x)
        # print(x_embed.size())
        sent_outs = []
        for i in range(self.doc_len):
            sent_outs.append(self.sent_wise_lstms[i](x_embed.transpose(0,1)[i]))
        sent_att_outs = []
        for i in range(self.doc_len):
            # print(self.sent_wise_attlstms[i](sent_outs[i]).unsqueeze(2), self.sent_wise_attlstms[i](sent_outs[i]).unsqueeze(2).size())
            sent_att_outs.append(self.sent_wise_attlstms[i](sent_outs[i]).unsqueeze(2))
        doc_representation = torch.cat(sent_att_outs, dim=2).transpose(1, 2)
        # print(doc_representation, doc_representation.size())
        doc_lstm = self.doc_lstm(doc_representation)
        att_doc_lstm = self.doc_attention(doc_lstm)
        y_ = self.linear_stack(att_doc_lstm)
        y=F.softmax(y_)
        return y
