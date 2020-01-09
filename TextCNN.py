# _*_coding:utf-8_*_
# user : yuling
# time : 2010-01-06 20:52
# info : TextCNN model  and TextRCNN model

import torch
from torch import nn
import torch.nn.functional as F

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]

class TextCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
            pretrained_embeddings,dropout=0.5):
        super(TextCNN,self).__init__()
        # self.embedding = nn.Embedding(len_vocab,100)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.convs = Conv1d(embedding_dim, n_filters, filter_sizes)  #一维卷积
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        text = x
        batch_size, seq_length = text.shape
        vec = self.embedding(text)
        #print(vec.shape)  #(batch_size,seq_length,embedding_dim)=(128,40,100)
        vec = vec.permute(0,2,1)
        #print(vec.shape)  #(128,100,40)  因为一维卷积是在最后的维度上扫的
        conved = self.convs(vec)
        #print([conv.shape for conv in conved])
        #(batch_size, n_filters, seq_length-filter_sizes[n]+1) = (128,100,40-2+1)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #print(pool.shape for pool in pooled)
        #(batch_size, n_filters)=(128,100)
        # cat函数将(A, B)，dim=0 按行拼接，dim=1按列拼接
        cat = self.dropout(torch.cat(pooled, dim=1))
        #print(cat.shape) #[128,300]
        out = self.fc(cat)
        #print(out.shape) #[128,2]
        return out

# RCNN region-CNN 中文名为区域卷积神经网络
class TextRCNN(nn.Module):   
    def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, pretrained_embeddings, dropout=0.5):
        super(TextRCNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True)
        #没有明白这个W2是个什么意思？？？
        self.W2 = nn.Linear(2*hidden_size + embedding_dim, hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        text = x
        batch_size, seq_length = text.shape
        embedding = self.dropout(self.embedding(text))
        embedding = embeding.permute(1, 0, 2)
        #print(embeding.shape)  #[seq_length, batch_size, embeding_dim]
        outputs = self.rnn(embedding)
        #outputs: [seq_lenth, batch_size, hidden_size*bidirectional]
        outputs = outputs.permute(1, 0, 2)
        embedding = embedding.permute(1, 0, 2)
        x.torch.cat((outputs, embedding),2)
        # x: [batch_size, seq_length, embedding_dim+hidden_size*bidirectional]
        y2 = torch.tanh(self.W2(x).permute(0, 2, 1))
        #y2: [batch_size, hidden_size*bidirectional, seq_length]
        y3 = torch.max_pool1d(y2, y2.size()[2]).squeeze(2)
        #y3: [batch_size, hidden_size*bidirectional]

        return self.fc(y3)













