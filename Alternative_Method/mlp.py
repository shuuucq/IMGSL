import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    accuracy_score
)
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--field', type=str, default='Database', help='specify the research field')
parser.add_argument('--path', type=str, default="data_origin", help='specify the data path')
parser.add_argument('--lr', type=float, default="0.01", help='learning rate')
args = parser.parse_args()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, enhance=False):
        super(MLP, self).__init__()
        self.enhance = enhance
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        if enhance:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x.cuda())
        if self.enhance:
            x = self.bn1(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.enhance:
            x = self.bn2(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = F.log_softmax(self.fc3(x), dim=1)
        return x

class cDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx]) if self.y is not None else self.x[idx]

    def __len__(self):
        return len(self.x)

def f1(precision, recall):
    return (2 * precision * recall) / (precision + recall) if precision and recall else 0

def train_and_evaluate(model, train_loader, val_loader, test_loader, device, n_epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    
    for epoch in range(n_epochs):
        model.train()
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.long().to(device)
            pred = model(x)
            weights = torch.FloatTensor([1, 1]).to(device)
            loss = F.nll_loss(pred, y, weight=weights)
            loss.backward()
            optimizer.step()
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.item()})

        # Validate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.long().to(device)
                pred = model(x)
                weights = torch.FloatTensor([1, 1]).to(device)
                loss = F.nll_loss(pred, y, weight=weights)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{n_epochs}], Validation Loss: {val_loss}')

    model.eval()
    y_real, y_pred, y_proba_minority = [], [], []
    with torch.no_grad():
        for data, y in test_loader:
            data, y = data.to(device), y.long().to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            y_real.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            y_proba_minority.append(output.cpu().numpy())
    return np.concatenate(y_real), np.concatenate(y_pred), np.concatenate(y_proba_minority)

def load_pickle_files(base_path, fold, data_type):
    """加载指定折的pickle文件"""
    file_name = f"{args.field}_{args.path}_fold_{fold + 1}_{data_type}.pkl"
    return joblib.load(os.path.join(base_path, file_name))

def main():
    device = 'cuda:0'
    base_save_dir = os.path.join("field/data_splits", args.path)

    folds = 10
    repetitions = 100
    Precision, Recall, ROC, PRC, F1_best, ACC = [], [], [], [], [], []
    Name = [[] for _ in range(1201)]

    for repetition in range(repetitions):
        for fold in range(folds):
            print(f"Processing repetition {repetition + 1}, fold {fold + 1}")

            train_data = load_pickle_files(base_save_dir, fold, "training")
            val_data = load_pickle_files(base_save_dir, fold, "validation")
            test_data = load_pickle_files(base_save_dir, fold, "testing")

            Xtrain, ytrain = train_data.drop(columns=['is_awarded', 'author_id']).values, train_data['is_awarded'].values
            Xval, yval = val_data.drop(columns=['is_awarded', 'author_id']).values, val_data['is_awarded'].values
            Xtest, ytest = test_data.drop(columns=['is_awarded', 'author_id']).values, test_data['is_awarded'].values

            train_dataset = cDataset(Xtrain, ytrain)
            val_dataset = cDataset(Xval, yval)
            test_dataset = cDataset(Xtest, ytest)

            train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

            model = MLP(input_dim=Xtrain.shape[1], hidden_dim=64, output_dim=2).to(device)
            y_real, y_pred, y_proba_minority = train_and_evaluate(model, train_loader, val_loader, test_loader, device)

            for idx in range(len(test_dataset)):
                Name[test_data['author_id'].values[idx]].append(y_pred[idx])

            roc = roc_auc_score(y_real, y_proba_minority[:, 1])
            prc = average_precision_score(y_real, y_proba_minority[:, 1], pos_label=1)
            precision, recall, thresholds = precision_recall_curve(y_real, y_proba_minority[:, 1], pos_label=1)
            recall_ = 0
            precision_ = 0
            Thresholds = 0
            f1_best = 0
            for pre_i in range(len(precision)):
                f = f1(precision[pre_i], recall[pre_i])
                if f > f1_best:
                    f1_best = f
                    Thresholds = thresholds[pre_i]
                    recall_ = recall[pre_i]
                    precision_ = precision[pre_i]

            F1_best.append(f1_best)
            y_pred = (y_proba_minority[:, 1] >= Thresholds).astype(int)
            acc = accuracy_score(y_real, y_pred)

            Precision.append(precision_)
            Recall.append(recall_)
            ROC.append(roc)
            PRC.append(prc)
            ACC.append(acc)

            print(f"Precision: {precision_}, Recall: {recall_}, F1_best: {f1_best}, ROC: {roc}, PRC: {prc}, ACC: {acc}")

    Stability = []
    for id_name in Name:
        if id_name:
            counts = np.bincount(id_name)
            stability = np.max(counts) / len(id_name)
            Stability.append(stability)
    Stability = np.array(Stability)
    print("Stability:", np.mean(Stability))

    # 保存结果和超参数到CSV文件
    header = ["Field", "Path", "Method", "Learning Rate", "Precision", "Recall", "F1_best", "F1_best_std", "ROC", "ROC_std", "PRC", "ACC","all_F1_best"]
    data = [
        args.field, args.path, 'mlp', args.lr,
        np.mean(Precision), np.mean(Recall), np.mean(F1_best), np.std(F1_best),
        np.mean(ROC), np.std(ROC), np.mean(PRC), np.mean(ACC),np.mean(F1_best)
    ]
    save_results_to_csv("results.csv", header, data)

    print("Results saved to results.csv")


def save_results_to_csv(filename, header, data):
    """将结果保存为CSV文件，使用逗号分隔，如果文件存在则追加内容"""
    if not os.path.exists(filename):
        df = pd.DataFrame([data], columns=header)
        df.to_csv(filename, sep=',', index=False)
    else:
        df = pd.DataFrame([data])
        df.to_csv(filename, sep=',', mode='a', header=False, index=False)

if __name__ == "__main__":
    main()
