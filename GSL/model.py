import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import global_max_pool
from utils import *
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, JumpingKnowledge
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from backbone_GSL import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GSL(torch.nn.Module):
    def __init__(self, args):
        super(GSL, self).__init__()
        self.args = args
        self.num_node_features = 7
        self.num_classes = args.num_classes
        self.conv_name = args.conv_name
        self.hidden_dim = args.hidden_dim
        self.IB_size = args.IB_size
        self.graph_metric_type = args.graph_metric_type
        self.graph_type = args.graph_type
        self.set_epsilon = args.set_epsilon
        self.tau=args.tau
        self.dataset_name=args.dataset_name
        self.PATH=args.PATH
        self.readout=args.readout
        self.beta=args.beta
        self.nhid=args.IB_size
        self.lamb=args.lamb
        self.dropout_ratio=args.dropout_ratio

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin2 = torch.nn.Linear(self.nhid // 2 + 1, self.nhid // 4)
        self.lin3 = torch.nn.Linear(self.nhid // 4, self.num_classes)


        if self.conv_name == "GCN":
            self.conv_name_gnn = myGCN(self.args, in_dim=self.num_node_features, out_dim=self.IB_size,
                                      hidden_dim=self.hidden_dim)
            self.motif_gnn = myGCN(self.args, in_dim=self.num_node_features, out_dim=self.IB_size,
                                      hidden_dim=self.hidden_dim)
        elif self.conv_name == "GIN":
            self.conv_name_gnn = myGIN(self.args, in_dim=self.num_node_features, out_dim=self.IB_size,
                                      hidden_dim=self.hidden_dim)
            self.motif_gnn = myGCN(self.args, in_dim=self.num_node_features, out_dim=self.IB_size,
                                      hidden_dim=self.hidden_dim)

        elif self.conv_name == "GAT":
            self.conv_name_gnn = myGAT(self.args, in_dim=self.num_node_features, out_dim=self.IB_size*2,
                                      hidden_dim=self.hidden_dim)

        self.graph_learner = GraphLearner(input_size=self.num_node_features, hidden_size=self.hidden_dim,
                                          graph_type=self.graph_type,set_epsilon=self.set_epsilon, 
                                          metric_type=self.graph_metric_type, 
                                          datasetname=self.dataset_name, PATH=self.PATH, device=None)


        if torch.cuda.is_available():
            self.conv_name_gnn = self.conv_name_gnn.cuda()
            self.motif_gnn = self.motif_gnn.cuda()
            self.graph_learner = self.graph_learner.cuda()
            self.lin1 = self.lin1.cuda()
            self.lin2 = self.lin2.cuda()
            self.lin3 = self.lin3.cuda()

    def __repr__(self):
        return self.__class__.__name__

    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def to(self, device):
        self.conv_name_gnn.to(device)
        self.motif_gnn. to(device)
        self.graph_learner.to(device)
        return self

    def reset_parameters(self):
        self.conv_name_gnn.reset_parameters()
        self.graph_learner.reset_parameters()
        self.motif_gnn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def learn_graph(self, node_features, graph_skip_conn=None, graph_include_self=True, init_adj=None):
        new_feature, new_adj = self.graph_learner(node_features)
        if new_adj.size(0) != init_adj.size(0) or new_adj.size(1) != init_adj.size(1):
            raise ValueError("new_adj and init_adj size mismatch: {} vs {}".format(new_adj.size(), init_adj.size()))  
        new_adj = new_adj * (init_adj != 0).float()        
        if graph_include_self:
            if torch.cuda.is_available():
                eye = torch.eye(new_adj.size(0)).cuda()
            else:
                eye = torch.eye(new_adj.size(0))
            new_adj = new_adj + eye  
        return new_feature, new_adj

    def cosine_similarity(self, a, b):
        return F.cosine_similarity(a, b, dim=-1)

    def infonce_loss1(self, positive_pairs, negative_pairs1, negative_pairs2, tau=0.3, epsilon=1e-8):
        pos_sim = self.cosine_similarity(positive_pairs[:, 0], positive_pairs[:, 1]) / tau
        neg_sim1 = self.cosine_similarity(positive_pairs[:, 0].unsqueeze(1), negative_pairs1) / tau
        neg_sim2 = self.cosine_similarity(positive_pairs[:, 1].unsqueeze(1), negative_pairs2) / tau
        if torch.any(pos_sim > 50) or torch.any(neg_sim1 > 50) or torch.any(neg_sim2 > 50):
            print("Warning: large values in sim, applying clamping")
            pos_sim = torch.clamp(pos_sim, min=-50, max=50)
            neg_sim1 = torch.clamp(neg_sim1, min=-50, max=50)
            neg_sim2 = torch.clamp(neg_sim2, min=-50, max=50)
        pos_sim = torch.exp(pos_sim)
        neg_sim1 = torch.exp(neg_sim1)
        neg_sim2 = torch.exp(neg_sim2)
        pos_sim = pos_sim / (pos_sim + neg_sim1.sum(dim=-1) + neg_sim2.sum(dim=-1) + epsilon)
        loss = -torch.log(pos_sim + epsilon)
        return loss.mean()
    
    def infonce_loss2(self, positive_pairs, negative_pairs, tau=0.3, epsilon=1e-8):
        pos_sim = self.cosine_similarity(positive_pairs[:, 0], positive_pairs[:, 1]) / tau
        neg_sim = self.cosine_similarity(positive_pairs[:, 0].unsqueeze(1), negative_pairs) / tau
        if torch.any(pos_sim > 50) or torch.any(neg_sim > 50) :
            print("Warning: large values in sim, applying clamping")
            pos_sim = torch.clamp(pos_sim, min=-50, max=50)
            neg_sim= torch.clamp(neg_sim, min=-50, max=50)
        pos_sim = torch.exp(pos_sim)
        neg_sim = torch.exp(neg_sim)
        pos_sim = pos_sim / (pos_sim + neg_sim.sum(dim=-1)  + epsilon)
        loss = -torch.log(pos_sim + epsilon)
        return loss.mean()
    
    def safe_to_dense_adj(self, edge_index, num_nodes):
        if edge_index.numel() == 0: 
            return torch.zeros((num_nodes, num_nodes), device=edge_index.device) 
        
        max_index = edge_index.max().item()  
        required_size = max(max_index + 1, num_nodes)  
        adj = to_dense_adj(edge_index, max_num_nodes=required_size)[0] 
        return adj
    
    def forward(self, graphs,paper_count):
        device = graphs.x.device
        num_sample = graphs.num_graphs
        graphs_list = graphs.to_data_list()
        new_graphs_list = []

        for graph in graphs_list:
            device = graph.x.device
            x, edge_index,motif_edge_index,motif_edge_count = graph.x.to(device), graph.edge_index.to(device),graph.motif_adj.to(device),graph.motif_edge_count.to(device)
            raw_adj = self.safe_to_dense_adj(edge_index, x.shape[0])
            new_feature, new_adj = self.learn_graph(node_features=x,
                                                    graph_skip_conn=self.args.graph_skip_conn,
                                                    graph_include_self=self.args.graph_include_self,
                                                    init_adj=raw_adj)
            
            new_edge_index, new_edge_attr = dense_to_sparse(new_adj)

            new_graph = Data(x=new_feature, edge_index=new_edge_index, edge_attr=new_edge_attr,motif_adj=motif_edge_index,motif_edge_count=motif_edge_count)
            new_graphs_list.append(new_graph)
        
        new_loader = DataLoader(new_graphs_list, batch_size=len(new_graphs_list), collate_fn=lambda x: Batch.from_data_list(x))
        origin_loader = DataLoader(graphs_list, batch_size=len(graphs_list), collate_fn=lambda x: Batch.from_data_list(x))

        new_batch_data = next(iter(new_loader))
        origin_batch_data = next(iter(origin_loader))

        new_embs, _ = self.conv_name_gnn(new_batch_data.x, new_batch_data.edge_index)
        if(self.PATH=='data_extend'):
            new_motif_embs,_=self.motif_gnn(new_batch_data.x,new_batch_data.motif_adj,new_batch_data.motif_edge_count)
            totall_new_embs = new_embs + new_motif_embs*self.lamb
        else:
             totall_new_embs = new_embs
        origin_embs, _ = self.conv_name_gnn(origin_batch_data.x, origin_batch_data.edge_index)

   
        if self.args.readout == "max":
            new_embs = global_max_pool(totall_new_embs, new_batch_data.batch)
            origin_embs = global_max_pool(origin_embs, new_batch_data.batch)

        elif self.args.readout == "mean":
            new_embs = global_mean_pool(totall_new_embs, new_batch_data.batch)
            origin_embs = global_mean_pool(origin_embs, new_batch_data.batch)

    
        positive_pairs = torch.stack([origin_embs, new_embs], dim=1) 
        negative_pairs = []
        for i in range(num_sample):
            neg_samples = torch.cat([origin_embs[:i], origin_embs[i+1:] ,new_embs[:i], new_embs[i+1:]], dim=0) 
            negative_pairs.append(neg_samples.unsqueeze(0))
        negative_pairs = torch.cat(negative_pairs, dim=0)  

        total_infonce_loss = self.infonce_loss2(positive_pairs, negative_pairs)
        x=new_embs
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)   
        paper_count = torch.unsqueeze(paper_count,1)    
        x = torch.cat((x, paper_count),1)
        x = F.relu(self.lin2(x))    
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return  x, total_infonce_loss, graphs_list, new_graphs_list

class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, graph_type, set_epsilon=None, metric_type="dot",
                 datasetname='NLP',PATH='data-extend',device=None):
        super(GraphLearner, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.graph_type = graph_type
        self.set_epsilon = set_epsilon
        self.metric_type = metric_type
        self.datasetname = datasetname
        self.PATH = PATH

      
        end_path = os.path.basename(self.PATH)
        if end_path == 'data_extend':
            if self.datasetname=='NLP' or self.datasetname=='SoftwareEngineering'or self.datasetname=='HCI' or self.datasetname=='IR':
                self.hidden_size = hidden_size//2
            else:
                self.hidden_size = hidden_size       

        elif end_path == 'data_origin': 
            self.hidden_size = hidden_size//2

        print(self.hidden_size)

        self.lin1 = nn.Linear(self.input_size, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, self.hidden_size//2)

       
        print('[ Graph Learner metric type: {}, Graph Type: {} ]'.format(metric_type, self.graph_type))


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)  

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, node_features):
        learned_adj = self.learn_adj(node_features)
        return node_features, learned_adj


    def learn_adj(self, context):
        if self.metric_type == 'dot':
            context_fc = F.relu(self.lin2(F.relu(self.lin1(context))))
            attention = torch.matmul(context_fc, context_fc.transpose(-1, -2))
            markoff_value = 0

        if self.graph_type == 'epsilonNN':
            attention = self.build_epsilon_neighbourhood(attention, self.set_epsilon, markoff_value)
        return attention

    def build_epsilon_neighbourhood(self, attention, set_epsilon, markoff_value):
        attention = torch.sigmoid(attention)
        epsilon = torch.quantile(attention, set_epsilon)
        mask = (attention >= epsilon).float() 
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

