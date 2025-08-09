
import torch
import torch.nn.functional as F
import os
import pickle
import csv
import logging
import networkx as nx
import matplotlib.pyplot as plt
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
from load_data import prepare_data_loaders
from evaluate import *
from utils import *
from model import GSL as GSL_motif

# seed_everything(6789)s

args = parameter_parser()
args = config_args(args)

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

end_path = os.path.basename(args.PATH)


if end_path == 'data_extend':
    data_path="../extend-motif-data"
elif end_path == 'data_origin':
    data_path="../origin-motif-data"

def cross_validation_with_val_set(model_class, args, logger=None):
    folds = args.folds
    epochs = args.epochs
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size 
    lr = args.lr
    lr_decay_factor = args.lr_decay_factor 
    lr_decay_step_size = args.lr_decay_step_size
    weight_decay = args.weight_decay

    log_dir = os.path.join(os.getcwd(), 'logs', args.dataset_name, args.PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f'{args.dataset_name}_training.log')
    logger = setup_logger('training_logger', log_filename)
    logger.info("Starting cross-validation")
    logger.info(args)  

    round = 10
    all_rounds_precision = []
    all_rounds_recall = []
    all_rounds_f1_05 = []
    all_rounds_f1_best = []
    all_rounds_roc = []
    all_rounds_prc = []
    all_rounds_acc = []
    all_f1_best = 0
 
    for r in range(round):
        Precision = []
        Recall = []
        ROC = []
        PRC = []
        F1_05 = []
        ACC = []
        F1_best=[]
        for fold in range(folds):
            logger.info(f"Processing fold {fold + 1}/{folds}")
            train_loader, val_loader, test_loader = prepare_data_loaders(data_path, args, fold, device)
            y_pred = []
            y_real = []
            y_proba_minority = []
            model = model_class(args)
            model.to(device).reset_parameters()
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for epoch in range(1, epochs + 1):
                if model.__repr__() in ['GF']:
                    train_acc, train_loss = train_GF(args, model, optimizer,train_loader)
                    val_acc, val_loss=validate_GF(args,model,val_loader)
                    test_acc, loss_test, pred_y, lables, probas = compute_test_GF(args,model,test_loader)
                
                    y_pred.append(pred_y)
                    y_real.append(lables)
                    y_proba_minority.append(probas)
                                
                elif model.__repr__() in ['GSL','GSL_motif','NON_GSL','NON_GSL_ADD_MOTIF']:
                    train_loss, train_acc = train_GSL(model, optimizer, train_loader, epoch)
                    val_loss, val_acc = eval_GSL_loss(model, val_loader)
                    test_acc, probas, lables, data, pred_y, graphs_list, new_graphs_list = eval_GSL_acc(model, test_loader)


                    y_pred.append(pred_y)
                    y_real.append(lables)
                    y_proba_minority.append(probas)

                elif model.__repr__() in ['VIBGSL']:
                    train_loss, train_acc = train_VGIB(model, optimizer, train_loader, epoch)
                    val_loss, val_acc = eval_VGIB_loss(model, val_loader)
                    test_acc, probas, lables, data, graphs_list, new_graphs_list, pred_y = eval_VGIB_acc(model, test_loader)

                    y_pred.append(pred_y)
                    y_real.append(lables)
                    y_proba_minority.append(probas)

                elif model.__repr__() in ['PAGERANK', 'HITS', "IDME"]:
                    train_loss, train_acc = train(model, optimizer, train_loader)
                    val_loss, val_acc = eval_loss(model, val_loader)
                    test_acc, probas, lables, data, pred_y = eval_acc(model, test_loader)

                    y_pred.append(pred_y)
                    y_real.append(lables)
                    y_proba_minority.append(probas)
                                
                else:
                    raise ValueError('Unknown model: {}'.format(model.__repr__()))

                if epoch % lr_decay_step_size == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_decay_factor * param_group['lr']
                                        
                if epoch % 10 == 0:
                    message="epoch:",epoch,"acc_train:",train_acc,"loss_train:",train_loss,"acc_val",val_acc,"loss_val:",val_loss,"acc_test:",test_acc
                    print(message)
                    logger.info(message)

        
            y_pred = np.concatenate(y_pred)
            y_real = np.concatenate(y_real)
            logger.info(y_pred)
            logger.info( y_real)
            y_proba_minority = np.concatenate(y_proba_minority)
            f1_05 = f1_score(y_real,y_proba_minority[:,1]>=0.5)  
            roc = roc_auc_score(y_real, y_proba_minority[:,1])
            prc = average_precision_score(
                y_real,
                y_proba_minority[:,1],
                pos_label=1,
            )
            precision, recall, thresholds = precision_recall_curve(
                y_real,
                y_proba_minority[:,1],
                pos_label=1,
            )
            recall_=0
            precision_=0
            Thresholds=0
            f1_best=0
            for pre_i in range(len(precision)):
                f=f1(precision[pre_i],recall[pre_i])
                if(f>f1_best):
                    f1_best=f
                    Thresholds=thresholds[pre_i]
                    recall_=recall[pre_i]
                    precision_=precision[pre_i]
            # if(f1_best>all_f1_best):
            #     all_f1_best=f1_best
            #     save_best_model_and_data( all_f1_best, model, train_loader, val_loader, test_loader, r, fold)

            F1_05.append(f1_05)
            F1_best.append(f1_best)
            y_pred = (y_proba_minority[:,1] >= Thresholds).astype(int)
            acc=accuracy_score(y_real,y_pred)

            for label_index, label in enumerate([1]):
                print(
                    "Overall Precision, Recall, F1_0.5, F1, ROC, PRC, ACC for Class",
                    label,
                    ": ",
                    precision_,
                    ", ",
                    recall_,
                    ", ",
                    f1_05,
                    ", ",
                    f1_best,
                    ", ",
                    roc,
                    ", ",
                    prc,
                    ", ",
                    acc
                )
            Precision.append(precision_)
            Recall.append(recall_)
            ROC.append(roc)
            PRC.append(prc)
            ACC.append(acc)
            # visualize_graphs(str(fold),args.dataset_name,args.PATH, graphs_list, new_graphs_list)

        
        all_rounds_precision.append(np.mean(Precision))
        all_rounds_recall.append(np.mean(Recall))
        all_rounds_f1_05.append(np.mean(F1_05))
        all_rounds_f1_best.append(np.mean(F1_best))
        all_rounds_roc.append(np.mean(ROC))
        all_rounds_prc.append(np.mean(PRC))
        all_rounds_acc.append(np.mean(ACC))
        print(F1_best)
        print(np.mean(Precision),np.mean(Recall),np.mean(F1_05),np.mean(F1_best),np.mean(ROC),np.mean(PRC),np.mean(ACC))
        print(all_rounds_f1_best)
        logger.info(all_rounds_f1_best)


    data=compute_and_log_results(args, all_rounds_precision, all_rounds_recall,all_rounds_f1_05, all_rounds_f1_best, all_rounds_roc, all_rounds_prc, all_rounds_acc)
    logger.info(data)


def main():
    args = parameter_parser()
    args = config_args(args)
    seed_everything(6789)
    print(args)

    model_name = args.model_name
    if(model_name=='GF'):
        model_class=GF
    elif(model_name=='GSL'):
        model_class=GSL
    elif(model_name=='VIBGSL'):
        model_class=VIBGSL
    elif(model_name=='PAGERANK'):
        model_class=PAGERANK
    elif(model_name=='HITS'):
        model_class=HITS
    elif(model_name=='IDME'):
        model_class=IDME
    elif(model_name=='GSL_motif'):
        model_class=GSL_motif
    elif(model_name=='non_GSL'):
        model_class=NON_GSL
    elif(model_name=='non_GSL+motif'):
         model_class=NON_GSL_ADD_MOTIF
    print(model_class)
    cross_validation_with_val_set(model_class, args, logger=None)


if __name__ == '__main__':
    main()