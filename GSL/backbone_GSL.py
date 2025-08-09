import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool,BatchNorm
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN


class GCN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(GCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
            x = F.relu(self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr))
            x = F.relu(self.conv2(x=x, edge_index=edge_index, edge_weight=edge_attr))
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class myGCN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(myGCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index,edge_count=None):
        if edge_count is not None:
            edge_count = edge_count.squeeze(1)  # 转换为 [num_edges] 形状
            edge_min, edge_max = edge_count.min(), edge_count.max()
            edge_weight = (edge_count - edge_min) / (edge_max - edge_min)          
            edge_weight = edge_weight.float()  # 确保是浮动类型
        else:
            edge_weight = None  # 如果没有提供 edge_count，则不使用边权重
        edge_index = edge_index.long()  # 确保是整数类型
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        node_embeddings = x

        return node_embeddings, F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GIN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(GIN, self).__init__()
        self.args = args
        self.conv1 = GINConv(
            Sequential(
                Linear(in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(self.args.num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        BN(hidden_dim),
                    ), train_eps=True))
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index,batch):
        
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class myGIN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(myGIN, self).__init__()
        self.args = args
        self.conv1 = GINConv(
            Sequential(
                Linear(in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            ), train_eps=True)
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, out_dim),
                ReLU(),
                BN(out_dim),
            ), train_eps=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_count=None): 
        if edge_count is not None:
            edge_count = edge_count.squeeze(1)  # 转换为 [num_edges] 形状
            mean_edge_count = edge_count.mean()  # 计算均值
            std_edge_count = edge_count.std()    # 计算标准差
            edge_weight = (edge_count - mean_edge_count) / std_edge_count  # 标准化
            edge_weight = edge_weight.float()  # 确保是浮动类型
        else:
            edge_weight = None  # 如果没有提供 edge_count，则不使用边权重
        edge_index = edge_index.long()  # 确保是整数类型
        x = self.conv1(x, edge_index,edge_weight)
        x = self.conv2(x, edge_index,edge_weight)
        node_embeddings = x
        return node_embeddings, F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__




class GAT(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(GAT, self).__init__()
        self.args = args
        self.conv1 = GATConv(in_dim, hidden_dim, heads=8, dropout=0.5)

        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=1,
                             concat=False, dropout=0.5)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
            x = F.relu(self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr))
            x = F.relu(self.conv2(x=x, edge_index=edge_index, edge_weight=edge_attr))
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class myGAT(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(myGAT, self).__init__()
        self.args = args
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, dropout=0.5)

        self.conv2 = GATConv(hidden_dim * 4, out_dim, heads=1,
                             concat=False, dropout=0.5)
        self.relu = torch.nn.LeakyReLU(0.2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        node_embeddings = x
        return node_embeddings, F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__