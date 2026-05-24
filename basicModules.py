from torch import nn as nn
import torch
from math import sqrt
import torch.nn.functional as F


class BatchNorm1d(nn.Module):
    def __init__(self, inSize, name='batchNorm1d'):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(inSize)
        self.name = name

    def forward(self, x):
        return self.bn(x)


class BnodeEmbedding(nn.Module):
    def __init__(self, embedding, dropout, freeze=False):
        super(BnodeEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(embedding, dtype=torch.float32).detach(),
            freeze=freeze
        )
        self.dropout1 = nn.Dropout(p=dropout / 2)
        self.dropout2 = nn.Dropout(p=dropout / 2)
        self.p = dropout

    def forward(self, x):
        if self.p > 0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x


class MLP(nn.Module):
    def __init__(self, inSize, outSize, dropout, actFunc,
                 outBn=True, outAct=False, outDp=False):
        super(MLP, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        x = self.out(x)
        if self.outBn:
            if len(x.shape) == 2:
                x = self.bns(x)
            else:
                x = self.bns(x.transpose(-1, -2)).transpose(-1, -2)
        if self.outAct:
            x = self.actFunc(x)
        if self.outDp:
            x = self.dropout(x)
        return x


class GCN(nn.Module):
    def __init__(self, inSize, outSize, dropout, layers, resnet, actFunc,
                 outBn=False, outAct=True, outDp=True):
        super(GCN, self).__init__()
        self.gcnlayers = layers
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.resnet = resnet

    def forward(self, x, L):
        m_all = x[:, 0, :].unsqueeze(1)
        d_all = x[:, 1, :].unsqueeze(1)
        for _ in range(self.gcnlayers):
            a = self.out(torch.matmul(L, x))
            if self.outBn:
                if len(L.shape) == 3:
                    a = self.bns(a.transpose(1, 2)).transpose(1, 2)
                else:
                    a = self.bns(a)
            if self.outAct:
                a = self.actFunc(a)
            if self.outDp:
                a = self.dropout(a)
            if self.resnet and a.shape == x.shape:
                a += x
            x = a
            m_all = torch.cat((m_all, x[:, 0, :].unsqueeze(1)), 1)
            d_all = torch.cat((d_all, x[:, 1, :].unsqueeze(1)), 1)
        return m_all, d_all


class LayerAtt(nn.Module):
    def __init__(self, inSize, outSize, gcnlayers):
        super(LayerAtt, self).__init__()
        self.layers = gcnlayers + 1
        self.q = nn.Linear(inSize, outSize)
        self.k = nn.Linear(inSize, outSize)
        self.v = nn.Linear(inSize, outSize)
        self.norm = 1 / sqrt(outSize)
        self.actfun1 = nn.Softmax(dim=1)
        self.attcnn = nn.Conv1d(
            in_channels=self.layers, out_channels=1,
            kernel_size=1, stride=1, bias=True
        )

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        att_scores = torch.bmm(Q, K.transpose(1, 2)) * self.norm
        alpha = self.actfun1(att_scores)
        z = torch.bmm(alpha, V)
        cnnz = self.attcnn(z.transpose(1, 2))
        finalz = cnnz.squeeze(dim=1)
        return finalz


class LayerAtt2(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()
        self.Q = nn.Linear(inSize, outSize)
        self.K = nn.Linear(inSize, outSize)
        self.V = nn.Linear(inSize, outSize)
        self.norm = 1. / (outSize ** 0.5)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        att = torch.matmul(Q, K.transpose(1, 2)) * self.norm
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, V)
        return out.mean(dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, inSize, outSize, gcnlayers, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [LayerAtt2(inSize, outSize) for _ in range(num_heads)]
        )
        self.merge_layer = nn.Linear(num_heads * outSize, outSize)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.merge_layer(concatenated)