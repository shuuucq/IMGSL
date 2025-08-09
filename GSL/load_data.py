import os
import pickle
import random
import torch
import numpy as np
from utils import *
from torch_geometric.loader import DataLoader

def load_data_from_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_data_loaders(data_path,args,fold,device):
    seed_everything(6789)
    train_dataset = load_data_from_pickle(os.path.join(data_path, f"{args.dataset_name}-{fold + 1}_training.pkl"))
    val_dataset = load_data_from_pickle(os.path.join(data_path, f"{args.dataset_name}-{fold + 1}_validation.pkl"))
    test_dataset = load_data_from_pickle(os.path.join(data_path, f"{args.dataset_name}-{fold + 1}_testing.pkl"))

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, args.test_batch_size, shuffle=False,worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, args.test_batch_size, shuffle=False,worker_init_fn=worker_init_fn)
    return train_loader, val_loader, test_loader

