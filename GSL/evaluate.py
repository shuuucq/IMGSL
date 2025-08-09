import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, average_precision_score, f1_score
import time
import math
import os
import pickle
import csv
import logging
import networkx as nx
import matplotlib.pyplot as plt
import re
from model import *
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Dataset, Batch
from torch_geometric.loader import DataLoader
from utils import *
from param_parser import parameter_parser
Name=[[] for i in range(1201)] #index by rank, pred_class as value

def preprocess_data(args,data):
    if args.ignore_edge:
        data.edge_index = torch.tensor([[] for _ in range(data.edge_index.shape[0])], dtype=torch.long)
        data.edge_attr = torch.tensor([[]], dtype=torch.long).t()
    if args.ignore_node_attr:
        data.x = torch.ones((data.x.shape[0], 1), dtype=data.x.dtype)
    return data

def train_GSL(model, optimizer, loader, epoch):
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_infonce_loss = 0
    correct = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        data.y = data.y.long()

        paper_count = [0]*(data.batch[data.batch.shape[0]-1]+1)    
        for j in range(data.batch.shape[0]):
            paper_count[data.batch[j]] += 1
        paper_count = Variable(torch.FloatTensor(paper_count)).to(device)

        logits, infonceloss,_,_ = model(data,paper_count)
        weights = np.array([1, 2])
        weights = torch.FloatTensor(weights).to(device)
        class_loss = F.nll_loss(logits, data.y, weight=weights)

        loss = class_loss + infonceloss * model.beta

        if loss.dim() > 0: 
            loss = loss.mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        total_loss += loss.item() * num_graphs(data)
        total_class_loss += class_loss.item() * num_graphs(data)
        total_infonce_loss += infonceloss.item() * num_graphs(data)
        optimizer.step()

        pred = logits.max(1)[1]

        correct += pred.eq(data.y.view(-1)).sum().item()
    
    avg_total_loss = total_loss / len(loader.dataset)
    avg_class_loss = total_class_loss / len(loader.dataset)
    avg_infonce_loss = total_infonce_loss / len(loader.dataset)
    
    return avg_total_loss, correct / len(loader.dataset)

def eval_GSL_acc(model, loader):
    model.eval()
    graphs_list = []
    new_graphs_list = []
    correct = 0.0
    
    for data in loader:
        data = data.to(device)  
        y = data.y.to(device)  
        paper_count = [0]*(data.batch[data.batch.shape[0]-1]+1)
        for j in range(data.batch.shape[0]):
            paper_count[data.batch[j]] += 1
        paper_count = Variable(torch.FloatTensor(paper_count)).to(device)
        with torch.no_grad():
            logits, infonceloss , graphs_list, new_graphs_list= model(data, paper_count)

            probs = logits.softmax(dim=1).cpu().numpy()
            pred=logits.max(dim=1)[1]
            correct += pred.eq(y).sum().item()
            preds = pred.cpu().long().numpy()
            labels = torch.squeeze(y).cpu()
            labels = labels.long().numpy()

    return correct / len(loader.dataset), probs, labels, data, preds, graphs_list, new_graphs_list



def eval_GSL_loss(model, loader):
    model.eval()
    total_loss = 0
    total_class_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        data.y = data.y.long()
        paper_count = [0]*(data.batch[data.batch.shape[0]-1]+1)
        for j in range(data.batch.shape[0]):
            paper_count[data.batch[j]] += 1
        paper_count = Variable(torch.FloatTensor(paper_count)).to(device)
        with torch.no_grad():
            logits, infonce_loss,_,_ = model(data,paper_count)
            pred = logits.max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
            weights = np.array([1, 2])
            weights = torch.FloatTensor(weights).to(device)
            class_loss = F.nll_loss(logits, data.y, weight=weights)
       
            loss = class_loss.item() + infonce_loss*model.beta
            total_loss += loss * num_graphs(data)
            total_class_loss += class_loss * num_graphs(data)
    return  total_loss / len(loader.dataset), correct / len(loader.dataset)


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

