from copy import deepcopy
from numbers import Number
from torch.autograd import Variable
from texttable import Texttable
from param_parser import parameter_parser
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, subgraph
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from param_parser import parameter_parser
import random
import os
import logging
import time
import math
import torch
import torch.nn.functional as F
import os
import pickle
import csv
import logging
import networkx as nx
import matplotlib.pyplot as plt
import re
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Dataset, Batch
from torch_geometric.loader import DataLoader
from param_parser import parameter_parser
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
 )
from evaluate import *
from utils import *
from model import *
VERY_SMALL_NUMBER = 1e-12

def seed_everything(seed=6789):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(6789)
def normalize_adj(mx): #计算对称化的归一化矩阵
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def print_dataset(dataset):
    num_nodes = num_edges = 0
    for data in dataset:
        num_nodes += data.num_nodes
        num_edges += data.num_edges

    print('Name', dataset)
    print('Graphs', len(dataset))
    print('Nodes', num_nodes / len(dataset))
    print('Edges', (num_edges // 2) / len(dataset))
    print('Features', dataset.num_features)
    print('Classes', dataset.num_classes)
    print()


def dropout(x, drop_prob, shared_axes=[], training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    input_tensor: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
    """
    if drop_prob == 0 or drop_prob == None or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask


def create_mask(x, N, device=None):
    if isinstance(x, torch.Tensor):
        x = x.data
    mask = np.zeros((len(x), N))
    for i in range(len(x)):
        mask[i, :x[i]] = 1
    return torch.Tensor(mask).to(device)


def to_data_list(x, edge_index, y, batch):
    idx_max = batch.max().item()
    data_list = []
    base_num = 0
    for graph_id in range(idx_max+1):
        node_idx = [i for i in range(len(batch)) if batch[i] == graph_id] #获取当前图graph_id的所有节点索引
        new_x = x[node_idx] #获取节点特征和边
        new_edge_index = subgraph(node_idx, edge_index)[0]
        new_edge_index = new_edge_index - base_num #调整边索引

        data = Data(x=new_x, edge_index=new_edge_index, y=[y[graph_id]]) 
        data_list.append(data)
        base_num += len(node_idx)
    return data_list


def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
            isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list

def save_best_model_and_data(all_f1_best, model, train_loader, val_loader, test_loader, r, fold):
   
    save_dir='save_data/'
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Paths for model and data
    model_save_path = os.path.join(save_dir, f'best_model_round_{r}_fold_{fold}.pth')
    train_data_path = os.path.join(save_dir, f'train_data_round_{r}_fold_{fold}.pth')
    val_data_path = os.path.join(save_dir, f'val_data_round_{r}_fold_{fold}.pth')
    test_data_path = os.path.join(save_dir, f'test_data_round_{r}_fold_{fold}.pth')

    # Save the best model state
    torch.save(model.state_dict(), model_save_path)

    # Save data
    save_data(train_loader, train_data_path)
    save_data(val_loader, val_data_path)
    save_data(test_loader, test_data_path)

    print(f"New best model saved with F1 score: {all_f1_best} for round {r}, fold {fold}")

def save_data(loader, filename):
    """
    Saves data from a DataLoader to a specified file using PyTorch's save function.
    
    Parameters:
    - loader (DataLoader): DataLoader whose data needs to be saved.
    - filename (str): Path where the data will be saved.
    """
    try:
        data_list = [data for data in loader]
        torch.save(data_list, filename)
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")

def config_args(args):
    if args.model_name == "GF":
        args.epochs = 50
        args.weight_decay = 0.01
        
        if args.PATH == "data_extend":
            if args.dataset_name in ["NLP", "HCI", "Database"]:
                args.batch_size = 64
            else:
                args.batch_size = 200
            
            if args.dataset_name == "DataMining":
                args.weight_decay = 0.0001
        else:
            args.batch_size = 200
    else:
        args.epochs = 200

    return args



def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()  # Clear existing handlers
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def load_data_from_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data     

def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

def compute_and_log_results(args, Precision, Recall,F1_05, F1_best, ROC, PRC, ACC):
    # Compute mean stability if applicable
    
    # Define the output CSV file name
    recordfile = f"{args.PATH}_{args.dataset_name}_results.csv"
    # Define the data dictionary to write to CSV
    results_data = {
        "Model Name": args.model_name,
        "Dataset Name": args.dataset_name,
        "PATH": args.PATH,
        "Learning Rate": args.lr,
        "IB_size":args.IB_size,
        "Hidden Units": args.hidden_dim,
        "Dropout Ratio": args.dropout_ratio,
        "Convolution Name": args.conv_name,
        "Number of Layers": args.num_layers,
        "Averaging": args.average,
        "Batch Size": args.batch_size,
        "Weight Decay": args.weight_decay,
        "set_epsilon":args.set_epsilon,
        "tau":args.tau,
        "beta":args.beta,
        "lamb":args.lamb,
        "Precision": np.mean(Precision),
        "Recall": np.mean(Recall),
        "F1_best": np.mean(F1_05),
        "F1_best_std": np.std(F1_05),
        "F1_best": np.mean(F1_best),
        "F1_best_std": np.std(F1_best),
        "ROC": np.mean(ROC),
        "ROC_std": np.std(ROC),
        "PRC": np.mean(PRC),
        "ACC": np.mean(ACC),
        "ACC_std": np.std(ACC),
        "ALL F1":F1_best,
    }

    # Check if file exists to avoid writing headers repeatedly
    file_exists = os.path.isfile(recordfile)

    # Write results to CSV file
    with open(recordfile, 'a', newline='') as csvfile:
        fieldnames = list(results_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(results_data)

    # Console output for verification
    print(f"Results saved to {recordfile}")
    for key, value in results_data.items():
        print(f"{key}: {value}")

    return  results_data

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)
def visualize_graphs(fold, datasetname, path, graphs_list, new_graphs_list, num_graphs=None, output_dir="graphs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if num_graphs is None:
        num_graphs = min(len(graphs_list), len(new_graphs_list))

    def draw_graph(G, pos, ax, node_colors, node_labels, node_mapping):
        edge_colors = []
        for edge in G.edges():
            if node_labels[node_mapping[edge[0]]] == node_labels[node_mapping[edge[1]]]:
                edge_colors.append('gray')
            else:
                edge_colors.append('gray')
        nx.draw(G, pos, ax=ax, with_labels=False, node_size=300, node_color=node_colors, edge_color=edge_colors)

    for i in range(num_graphs):
        if i >= len(graphs_list) or i >= len(new_graphs_list):
            break
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        original_graph = graphs_list[i]
        new_graph = new_graphs_list[i]

        G_original = nx.Graph()
        G_original.add_nodes_from(range(original_graph.x.size(0)))  # 添加所有节点
        for edge in original_graph.edge_index.t().tolist():
            if edge[0] != edge[1]: 
                G_original.add_edge(edge[0], edge[1])
        
        G_new = nx.Graph()
        G_new.add_nodes_from(range(new_graph.x.size(0)))  # 添加所有节点
        for edge in new_graph.edge_index.t().tolist():
            if edge[0] != edge[1]: 
                G_new.add_edge(edge[0], edge[1])

        # 删除无连边的节点
        G_original.remove_nodes_from(list(nx.isolates(G_original)))
        G_new.remove_nodes_from(list(nx.isolates(G_new)))

        # 获取当前图的节点索引
        original_indices = list(G_original.nodes)
        new_indices = list(G_new.nodes)

        # 使用相同的布局并拉大节点之间的距离
        pos_original = nx.spring_layout(G_original, k=1.5, iterations=50)
        pos_new = {node: pos_original[node] for node in G_new.nodes() if node in pos_original}
        
        new_correct = "correct" if new_graph['true_label'] == new_graph['prediction'] else "incorrect"
        origin_correct = "correct" if original_graph['true_label'] == original_graph['prediction'] else "incorrect"
        
        # 从图对象中提取实际的节点数和边数
        original_node_count = G_original.number_of_nodes()
        original_edge_count = G_original.number_of_edges()
        new_node_count = G_new.number_of_nodes()
        new_edge_count = G_new.number_of_edges()

        # 确保颜色列表与节点数匹配
        original_node_labels = [original_graph.x[node, -1].item() for node in original_indices]
        new_node_labels = [new_graph.x[node, -1].item() for node in new_indices]

        # new_colors = ['#8FC3E4' if label == 0 else '#1D4C7C' for label in new_node_labels]  # Lime Green for label 0, Violet for label 1
        # original_colors = ['#F28034' if label == 0 else '#F8B94D' for label in original_node_labels]  # Lemon Yellow for label 0, Tomato Red for label 1
        original_colors = ['#6FC8CA' if label == 0 else '#3492B2' for label in original_node_labels]  # Lime Green for label 0, Violet for label 1
        new_colors = ['#58B8D1' if label == 0 else '#367DB0' for label in new_node_labels]  # Lemon Yellow for label 0, Tomato Red for label 1

        # 统计有连边的 label=1 和 label=0 的节点个数
        original_label_0_count = sum(1 for label in original_node_labels if label == 0)
        original_label_1_count = sum(1 for label in original_node_labels if label == 1)
        new_label_0_count = sum(1 for label in new_node_labels if label == 0)
        new_label_1_count = sum(1 for label in new_node_labels if label == 1)

        # 获取图的名称
        graph_name = extract_number(original_graph.name) if 'name' in original_graph else f'graph_{i+1}'

        ax = axes[0]
        draw_graph(G_original, pos_original, ax, original_colors, original_node_labels, {node: idx for idx, node in enumerate(original_indices)})
        ax.set_title(f'Initial Graph: {graph_name}\nNodes: {original_node_count}, Edges: {original_edge_count} \n'
                     f'Original Nodes: {original_label_0_count}, Extended Nodes: {original_label_1_count} \n'
                     f'Label: {original_graph["true_label"]}, Prediction: {original_graph["prediction"]}', fontsize=16)

        ax = axes[1]
        draw_graph(G_new, pos_new, ax, new_colors, new_node_labels, {node: idx for idx, node in enumerate(new_indices)})
        ax.set_title(f'Learned Graph: {graph_name}\nNodes: {new_node_count}, Edges: {new_edge_count} \n'
                     f'Original Nodes: {new_label_0_count}, Extended Nodes: {new_label_1_count} \n'
                     f'Label: {new_graph["true_label"]}, Prediction: {new_graph["prediction"]}', fontsize=16)

        plt.tight_layout()
       
        output_file = os.path.join(output_dir, datasetname, path, f'{fold}-graph_{i+1}_{origin_correct}_{new_correct}.pdf')
        
        # 确保路径存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close(fig)
        print(f"Saved graph {i+1} to {output_file}")



def extract_number(name):
    # 检查name是否为tensor类型，并提取数值
    if isinstance(name, torch.Tensor):
        name = name.item()
    match = re.findall(r'\d+', str(name))
    return match[0] if match else 'Graph'

def f1(precision, recall):
    if precision == 0 or recall == 0:
        return 1e-10  # 返回一个极小的非零值，避免 F1 分数为零
    return (2 * precision * recall) / (precision + recall)
def print_dataset_info(loader, loader_name="Loader"):
    print(f"Details of {loader_name}:")
    print(f"  Number of batches: {len(loader)}")
    # Attempt to print details from the first batch
    for batch in loader:
        if hasattr(batch, 'batch'):
            print(f"  Number of graphs in the first batch: {batch.num_graphs}")
        if hasattr(batch, 'x'):
            print(f"  Feature matrix (x) of the first graph in the first batch shape: {batch.x.size()}")
        if hasattr(batch, 'edge_index'):
            print(f"  Edge indices of the first graph in the first batch shape: {batch.edge_index.size()}")
        if hasattr(batch, 'y'):
            print(f"  Labels (y) of the first graph in the first batch: {batch.y.size() if batch.y.dim() > 0 else 'Scalar label: ' + str(batch.y)}")
        break  # Only process the first batch for demonstration purposes
