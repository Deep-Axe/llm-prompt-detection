"""Sentence CNN and bidirectional LSTM with padding-aware pooling."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentenceCNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim=64,filters=64,widths=(3,4,5)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim,filters,k) for k in widths])
        self.widths = widths
        self.head = nn.Linear(filters*len(widths),1)

    def forward(self,ids,lengths):
        x=self.embedding(ids).transpose(1,2)
        pooled=[]
        for k,conv in zip(self.widths,self.convs):
            z=torch.relu(conv(x))
            valid=torch.arange(z.shape[2],device=z.device)[None,:] < (lengths-k+1).clamp(min=1)[:,None]
            pooled.append(z.masked_fill(~valid[:,None,:],float('-inf')).amax(dim=2))
        return self.head(torch.cat(pooled,dim=1)).squeeze(1)


class BiLSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim=64,hidden_size=64):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.lstm=nn.LSTM(embedding_dim,hidden_size,batch_first=True,bidirectional=True)
        self.head=nn.Linear(hidden_size*4,1)

    def forward(self,ids,lengths):
        packed=pack_padded_sequence(self.embedding(ids),lengths.cpu(),batch_first=True,enforce_sorted=False)
        states,_=self.lstm(packed)
        states,_=pad_packed_sequence(states,batch_first=True,total_length=ids.shape[1])
        valid=torch.arange(ids.shape[1],device=ids.device)[None,:]<lengths[:,None]
        mean=(states*valid[:,:,None]).sum(1)/lengths[:,None]
        maximum=states.masked_fill(~valid[:,:,None],float('-inf')).amax(1)
        return self.head(torch.cat([mean,maximum],dim=1)).squeeze(1)
