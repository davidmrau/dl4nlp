import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils

class GRU(nn.Module):

    def __init__(self, batch_size,embedding_size, vocabulary_size, num_langs,gru_num_hidden=256,gru_num_layers=2, dropout_keep_prob=1.0):

        super(GRU, self).__init__()
        self.gru_num_hidden = gru_num_hidden
        self.gru_num_layers = gru_num_layers
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocabulary_size,embedding_size)
        self.gru = nn.GRU(embedding_size, gru_num_hidden, gru_num_layers)
        self.linear = nn.Linear(gru_num_hidden, num_langs)
        self.dropout = nn.Dropout(1-dropout_keep_prob)

    def forward(self, x, h=None):
        x = self.embedding(x)
        if h is not None:
            gru, hidden = self.gru(x, h)
        else:
            gru, hidden = self.gru(x)
        out = self.dropout(gru)
        out = self.linear(out)
        return out, hidden
