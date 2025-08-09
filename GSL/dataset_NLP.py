import torch
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import numpy as np
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
import os
import re
import pandas as pd
import joblib
import scipy as sc
def node_iter(G):
   return G.nodes

def node_dict(G):
    node_dict = G.nodes
    return node_dict

def read_graphfile(PATH):
    sub_graph_num=1000
    adj_list={i:[] for i in range(1,sub_graph_num+1)}     
    index_graph={i:[] for i in range(1,sub_graph_num+1)} 
    node_attrs={i:[] for i in range(1,sub_graph_num+1)}  
    graph_labels=[]
    graph_hindex=[]
    edge_weight=[]
    edge_proba=[]
    Name=[]
    authors_attributes=[]
    num_edges = 0
    index_i = 1
    for_i = -1
    for root, dirs, files in os.walk(PATH, topdown=False):
        for name in files:
            if(name[0]=='p'):
                for_i += 1
                path=PATH+'/'+name
                idx_features_labels = np.genfromtxt("{}".format(path),
                                    dtype=np.dtype(str))
                # build graph
                idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
                
                idx_map = {j: i for i, j in enumerate(idx)} 
                edges_unordered = np.genfromtxt(path.replace("papers", "influence"),
                                                dtype=np.dtype(str))\
                if(edges_unordered.shape[0]==0):
                    adj_list[index_i]=[]
                    edge_weight.append([])
                else:
                    if(len(edges_unordered.shape)==1):
                        edges_unordered=edges_unordered[np.newaxis,:]
                    edgeset=set()
                    edges_=[]
                    for j in range(edges_unordered.shape[0]):
                        if((edges_unordered[j][0],edges_unordered[j][1]) in edgeset or (edges_unordered[j][1],edges_unordered[j][0]) in edgeset):
                            continue
                        else:
                            if(edges_unordered[j][0]==edges_unordered[j][1]):
                                continue
                            edgeset.add((edges_unordered[j][0],edges_unordered[j][1]))
                            edges_.append(edges_unordered[j])
                            print(edges_unordered[j])
                    edges_unordered=np.array(edges_)
                    if(edges_unordered.shape[0]==0):
                        adj_list[index_i]=[]
                        edge_weight.append([])
                    else:
                        edge_w_temp = edges_unordered[:,2:].astype(np.float32) 
                        edge_w=[]
                        for j in range(edge_w_temp.shape[0]):
                            edge_w.append(edge_w_temp[j,:])
                            edge_w.append(edge_w_temp[j,:])
                        edge_w=np.array(edge_w).astype(np.float32)
                        edges_unordered = edges_unordered[:,:2] 
                        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                                        dtype=np.int32).reshape(edges_unordered.shape)
                        edge_weight.append(edge_w)
                        for line in edges:
                            e0,e1=(int(line[1]),int(line[0])) 
                            adj_list[index_i].append([e0,e1])
                            adj_list[index_i].append([e1,e0])
                            num_edges += 1

                fea_labels=pd.DataFrame(idx_features_labels)
                fea_labels[1]=pd.to_numeric(fea_labels[1])-1965
                fea_labels=fea_labels[[0,1,3,4,6,8,9]]
                idx_features_labels=fea_labels.values  
                for line in idx_features_labels[:, 1:]:
                    attrs = [float(attr) for attr in line]
                    node_attrs[index_i].append(np.array(attrs))

                authors = pd.read_csv("../run_data/NLP_data/csv/top_field_authors.csv", header = None)
                number=re.findall(r"\d+\d*", name)[0]
                number=int(number)

                if("2:" in authors[authors[10]==number][12].values[0] or "3:" in authors[authors[10]==number][12].values[0] or ("1:" in authors[authors[10]==number][12].values[0] and authors[authors[10]==number][5].values[0]>=3000)):
                    templist=[]
                    templist.append(1)
                    graph_labels.append(templist)
                else:
                    templist=[]
                    templist.append(0)
                    graph_labels.append(templist) 

                Name.append(name)
                index_i+=1
    return edge_weight,graph_labels,adj_list,node_attrs,Name
import re
class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, PATH, transform=None, union_field=False, pre_transform=None):
        self.PATH = PATH
        self.union_field = union_field
        transform=T.Compose([
        ])
        super(MultiSessionsGraph, self).__init__(root, transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return ['data.txt']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    def download(self):
        pass
    def process(self):
        data_list = []
        if self.union_field:
            data, slices = self.collate(self.union_field)
            torch.save((data, slices), self.processed_paths[0])
            print('union mode complete with data_attr_length:',len(data))
            return
        else:
            edge_weight,y,adj_list,node_attrs,Name=read_graphfile(PATH=self.PATH)
        round = len(y)
        iterbox=range(round)
        for i in iterbox:
            number = re.findall(r"\d+\d*",Name[i])
            number=int(number[0])
            nodeX=torch.tensor(np.array(node_attrs[i+1]),dtype=torch.float)
            pyg_graph=Data(x=nodeX, y=torch.tensor(y[i],dtype=torch.float), edge_index=torch.reshape(torch.tensor(adj_list[i+1],dtype=torch.long).t(),(2,-1)),edge_attr=torch.tensor(edge_weight[i],dtype=torch.float).reshape(-1,1),name=torch.tensor(number,dtype=torch.int))
            data_list.append(pyg_graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('one field mode complete with data_length:',len(data))