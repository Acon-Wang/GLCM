import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import math
from torch.nn.parameter import Parameter
import embedding
import embedding_5dim
import embedding_20dim

d_model = 600
n_head = 10
d_word = 150
d_char = 150
batch_size = 16
dropout = 0.1
dropout_char = 0.1

d_k = d_model // n_head
d_cq = d_model * 4
len_c = 450
len_q = 450

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.a = 1 / math.sqrt(d_k)

    def forward(self, x):
        bs, _, l_x = x.size()
        x = x.transpose(1, 2)

        k = self.k_linear(x).view(bs, l_x, n_head, d_k)
        q = self.q_linear(x).view(bs, l_x, n_head, d_k)
        v = self.v_linear(x).view(bs, l_x, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(bs * n_head, l_x, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs * n_head, l_x, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs * n_head, l_x, d_k)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.a

        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)
        out = out.view(n_head, bs, l_x, d_k).permute(1, 2, 0, 3).contiguous().view(bs, l_x, d_model)
        out = self.fc(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):#16
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.embed_1 = embedding.Embedding()
        self.embed_2 = embedding_5dim.Embedding()
        self.embed_3 = embedding_20dim.Embedding()

        self.block_1 = nn.LSTM(input_size=24*35, hidden_size=300, num_layers=2, batch_first=True,
                               bidirectional=True, dropout = 0.3)

        self.lc = nn.Conv1d(in_channels = 600, out_channels = 600, kernel_size=1)
        self.ma =  MultiHeadAttention()
        self.dwc = DepthwiseSeparableConv(600,64,3)
        self.eca = eca_layer(64)
        self.CBAM= CBAM(64)#4
        self.classifier = nn.Sequential(
            nn.Linear(333*64*2, 1, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, X):

        X_projected = torch.cat((self.embed_1(X), self.embed_2(X), self.embed_3(X)),-1)

        batch, _, _ = X_projected.size()

        out_LSTM, _ = self.block_1(X_projected.float())
        out = out_LSTM.transpose(1, 2)
        out1 = self.lc(out)
        out2 = self.ma(out)
        out = torch.cat((out1,out2), -1)
        out = self.dwc(out)
        residual = out
        out = out.unsqueeze(3)
        out = self.CBAM(out)  #引入卷积块注意力
        out = out.squeeze(3)
        out +=residual #残差结构
        out1 = out.contiguous().view(batch, -1)
        output = self.classifier(out1)
        return output

