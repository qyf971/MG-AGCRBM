import math
import torch
import torch.nn as nn
from GAT import GAT, GCN
import numpy as np
from mamba_ssm import Mamba
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))  # Matrix multiplication
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GRULayer(nn.Module):
    def __init__(self, in_channels, embed_size):
        super(GRULayer, self).__init__()

        self.short_gru = nn.GRU(in_channels,
                                embed_size,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=False)

        self.long_gru = nn.GRU(in_channels,
                               embed_size,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=False)

    def forward(self, x):
        [B, N, t, d] = x.shape
        x = x.reshape(-1, t, d)

        x_short = x[:, 24:, :]  # 近期数据
        x_long = x[:, :24, :]  # 周期数据
        x_long_week = x_long[:, :12, :]
        x_long_day = x_long[:, 12:, :]

        # 对于recent time使用short——gru
        x_short, _ = self.short_gru(x_short)
        # 对周周期与日周期使用long--gru
        x_long_week, _ = self.long_gru(x_long_week)
        x_long_day, _ = self.long_gru(x_long_day)

        x_long = torch.cat((x_long_week, x_long_day), dim=1)
        output_gru_layer = torch.cat((x_long, x_short), dim=1)
        output_gru_layer = output_gru_layer.reshape(B, N, t, -1)

        return output_gru_layer


class PredictionLayer(nn.Module):
    def __init__(self, T_dim, output_T_dim, embed_size):
        super(PredictionLayer, self).__init__()

        # 缩小时间维度。
        self.conv1 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, input_prediction_layer):
        out = self.relu(self.conv1(input_prediction_layer))  # 等号左边 out shape: [B, T, N, d]
        out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, d, N, T]
        out = self.conv2(out)  # 等号左边 out shape: [B, 1, N, T]
        out = out.squeeze(1)

        return out

class BidirectionalMamba(nn.Module):
    def __init__(self, embed_dim):
        """
        双向 Mamba（前向 + 反向相加）
        Args:
            embed_dim: 输入与输出的特征维度
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.forward_mamba = Mamba(d_model=embed_dim)
        self.backward_mamba = Mamba(d_model=embed_dim)

    def forward(self, x):
        """
        x: [B, T, D]
        输出: [B, T, D]
        """
        # 前向
        y_forward = self.forward_mamba(x)

        # 反向（翻转时间维度）
        x_rev = torch.flip(x, dims=[1])
        y_backward = self.backward_mamba(x_rev)
        y_backward = torch.flip(y_backward, dims=[1])

        # 相加融合
        y = y_forward + y_backward
        return y

class GCN_Layer(nn.Module):
    def __init__(self, device, adj, in_features, out_features):
        super(GCN_Layer, self).__init__()
        self.gcn_layer = GCN(device, adj, in_features, out_features)

    def forward(self, x):
        out = []
        for t in range(x.shape[2]):
            x_t = x[:, :, t, :]
            x_t = self.gcn_layer(x_t)
            out.append(x_t)
        out = torch.stack(out, dim=2)
        return out
        

class Model(nn.Module):
    def __init__(
            self,
            adj,
            in_channels,
            embed_size,
            time_num,
            T_dim,
            output_T_dim,
            heads,
            forward_expansion,
            dropout,
            num_nodes,
            apt_size,
            device):
        super(Model, self).__init__()

        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.forward_expansion = forward_expansion

        self.gru_layer = GRULayer(in_channels, embed_size)

        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, apt_size).to(device), requires_grad=True).to(device)  # (num_nodes,apt_size)
        self.node_vec2 = nn.Parameter(torch.randn(apt_size, num_nodes).to(device), requires_grad=True).to(device)  # (apt_size,num_nodes)

        self.gcn_layer = gcn(c_in=embed_size, c_out=embed_size, dropout=dropout, support_len=1)

        # self.gcn_layer = GCN_Layer(device, adj, embed_size, embed_size)

        self.bi_mamba = BidirectionalMamba(embed_size)

        # self.mamba = Mamba(embed_size)

        self.prediction_layer = PredictionLayer(T_dim, output_T_dim, embed_size)

    def forward(self, x, time_features):
        x = x.to(self.device)
        x = x.permute(0, 2, 3, 1)  # [B, N, t, 1]

        input_spatial_block = self.gru_layer(x)

        adaptive_adj = F.softmax(F.relu(torch.mm(self.node_vec1, self.node_vec2)), dim=1) # 自适应邻接矩阵

        output_spatial_block = self.norm1(self.gcn_layer(input_spatial_block.permute(0, 3, 1, 2), [adaptive_adj]).permute(0, 2, 3, 1) + input_spatial_block)

        [B, N, t, d] = output_spatial_block.shape
        output_spatial_block = output_spatial_block.reshape(-1, t, d)

        output_mamba = self.dropout(self.norm2(self.bi_mamba(output_spatial_block) + output_spatial_block))

        output_mamba = output_mamba.reshape(B, N, t, d)

        input_prediction_layer = output_mamba.permute(0, 2, 1, 3)
        out = self.prediction_layer(input_prediction_layer)

        return out  # [B, N, T]




