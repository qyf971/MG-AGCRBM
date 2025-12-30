import torch
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, Batch
import torch.nn as nn


# 将邻接矩阵转为PyG要求格式
def convert_adj(adj):
    data_adj = [[],[]]
    data_edge_features = []
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i,j] != 0:
                data_edge_features.append(adj[i,j])
                data_adj[0].append(i)
                data_adj[1].append(j)
    data_adj = torch.tensor(data_adj, dtype=torch.int64)
    data_edge_features = torch.Tensor(data_edge_features)
    return data_adj, data_edge_features


class GAT(torch.nn.Module):
    def __init__(self, device, in_features, out_features, edge_dim=1):
        super(GAT, self).__init__()
        self.gat = GATConv(in_features, out_features, edge_dim=edge_dim).to(device)
        self.convert = convert_adj
        self.device = device

    def forward(self, data, adj):
        adj = adj.cpu()
        data = data.cpu()
        data_adj, data_edge_features = self.convert(adj)
        data_list = [Data(x=x_, edge_index=data_adj, edge_attr=data_edge_features) for x_ in data]
        batch = Batch.from_data_list(data_list)
        batch.to(self.device)
        x, attention_weights = self.gat(batch.x, batch.edge_index, batch.edge_attr, return_attention_weights=True)
        x = x.view(len(data), len(adj), -1)
        return x, attention_weights

class GCN(nn.Module):
    def __init__(self, device, adj, in_features, out_features):
        """
        高效两层GCN，支持固定邻接矩阵（适用于交通流量、辐照度、空气质量等时空预测任务）
        Args:
            device: 运行设备 (cuda 或 cpu)
            adj: 邻接矩阵 (N, N)
            in_features: 输入维度
            out_features: 输出维度
            edge_dim: 边特征维度 (默认1)
            convert_adj: 邻接矩阵转换函数
        """
        super(GCN, self).__init__()
        self.device = device
        self.gcn1 = GCNConv(in_features, out_features).to(device)
        self.gcn2 = GCNConv(out_features, out_features).to(device)

        # ✅ 一次性预处理邻接矩阵
        data_adj, data_edge_features = convert_adj(adj)

        # ✅ 注册为buffer，避免反复移动设备
        self.register_buffer("edge_index", data_adj.to(torch.long))
        self.register_buffer("edge_attr", data_edge_features.to(torch.float32))

    def forward(self, data):
        """
        Args:
            data: [B, N, F] 张量
        Returns:
            out: [B, N, F_out]
        """
        B, N, F = data.shape

        # ✅ 构建批次图，只在forward中拼接x
        data_list = [Data(x=x_, edge_index=self.edge_index, edge_attr=self.edge_attr) for x_ in data]
        batch = Batch.from_data_list(data_list).to(self.device)

        # ✅ 图卷积
        x = self.gcn1(batch.x, batch.edge_index, batch.edge_attr)
        x = torch.relu(x)
        x = self.gcn2(x, batch.edge_index, batch.edge_attr)
        x = torch.relu(x)

        # ✅ 恢复为 [B, N, F_out]
        x = x.view(B, N, -1)
        return x

