import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import pickle
import os
import argparse
from sklearn import svm
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)
import joblib
import csv
from xgboost import XGBClassifier
import logging
from datetime import datetime

# Create a log directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Get current date for log file naming
current_date = datetime.now().strftime('%Y-%m-%d')
log_file = os.path.join(log_dir, f"script_{current_date}.log")

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will still print logs to the console
    ]
)

logging.info("Logging setup complete.")

# Command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--field', type=str, default='NLP', help='specify the research field')
parser.add_argument('--path', type=str, default="data_origin", help='specify the data path')
parser.add_argument('--lr', type=float, default=0.01, help='specify the learning rate')
args = parser.parse_args()

def load_pickle_files(base_path, fold, data_type):

    file_name = f"{args.field}_{args.path}_fold_{fold + 1}_{data_type}.pkl"
    return joblib.load(os.path.join(base_path, file_name))


def save_results_to_csv(filename, header, data):

    if not os.path.exists(filename):
        df = pd.DataFrame([data], columns=header)
        df.to_csv(filename, sep=',', index=False)
    else:
        df = pd.DataFrame([data])
        df.to_csv(filename, sep=',', mode='a', header=False, index=False)

def f1(precision, recall):
    if precision == 0 or recall == 0:
        return 0
    return (recall * precision * 2) / (recall + precision)

def main():
    base_save_dir = f"field/data_splits/{args.path}"

    folds = 10
    repetitions = 100

    Precision = []
    Recall = []
    ROC = []
    PRC = []
    F1_best = []
    ACC = []

    for repetition in range(repetitions):

        y_pred = []
        y_real = []
        y_proba_minority = []

        for fold in range(folds):
            # logging.info(f"Processing repetition {repetition + 1}, fold {fold + 1}")

            # Load the data for each fold
            train_data = load_pickle_files(base_save_dir, fold, "training")
            val_data = load_pickle_files(base_save_dir, fold, "validation")
            test_data = load_pickle_files(base_save_dir, fold, "testing")

            NameXtrain, NameXval, NameXtest = train_data['author_id'], val_data['author_id'], test_data['author_id']

            Xtrain, ytrain = train_data.drop(columns=['is_awarded', 'author_id']).values, train_data['is_awarded'].values
            Xval, yval = val_data.drop(columns=['is_awarded', 'author_id']).values, val_data['is_awarded'].values
            Xtest, ytest = test_data.drop(columns=['is_awarded', 'author_id']).values, test_data['is_awarded'].values

            # Standardization of the data
            Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0)
            Xval = (Xval - Xval.mean(axis=0)) / Xval.std(axis=0)
            Xtest = (Xtest - Xtest.mean(axis=0)) / Xtest.std(axis=0)

            # Grid search for best hyperparameters
            if repetition == 0 and fold == 0:
                parameters = {
                    'max_depth': range (2, 10, 1),
                    'n_estimators': range(60, 220, 40),
                }
                # Initialize XGBClassifier model
                model = XGBClassifier(learning_rate=0.05, n_estimators=60, objective='binary:logistic',
                        silent=True, max_depth=2, nthread=4, scale_pos_weight=2)

                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=parameters,
                    scoring='f1',
                    n_jobs=10,
                    cv=10,
                    verbose=True
                )
                grid_search.fit(Xtrain, ytrain)
                logging.info(f"Best parameters from grid search: {grid_search.best_params_}")
                best_params = grid_search.best_params_
            else:
                best_params = grid_search.best_params_
            
            # Train the classifier with the best parameters
            clf = XGBClassifier(**best_params)
            clf.fit(Xtrain, ytrain)

            # Validation set prediction and evaluation
            val_pred_proba = clf.predict_proba(Xval)
            val_pred = clf.predict(Xval)

            # Test set prediction and evaluation
            test_pred_proba = clf.predict_proba(Xtest)
            test_pred = clf.predict(Xtest)

            y_real.append(ytest)
            y_pred.append(test_pred)
            y_proba_minority.append(test_pred_proba)

        # Concatenate predictions for all folds
        y_pred = np.concatenate(y_pred)
        y_real = np.concatenate(y_real)
        y_proba_minority = np.concatenate(y_proba_minority)

        # Calculate performance metrics
        roc = roc_auc_score(y_real, y_proba_minority[:, 1])
        prc = average_precision_score(y_real, y_proba_minority[:, 1], pos_label=1)

        precision, recall, thresholds = precision_recall_curve(y_real, y_proba_minority[:, 1], pos_label=1)
        recall_ = 0
        precision_ = 0
        Thresholds = 0
        f1_best = 0

        # Find the best F1 score and corresponding threshold
        for pre_i in range(len(precision)):
            f = f1(precision[pre_i], recall[pre_i])
            if f > f1_best:
                f1_best = f
                Thresholds = thresholds[pre_i]
                recall_ = recall[pre_i]
                precision_ = precision[pre_i]

        # Calculate accuracy with the best threshold
        y_pred = (y_proba_minority[:, 1] >= Thresholds).astype(int)
        acc = accuracy_score(y_real, y_pred)

        # Log the results for each class
        for label_index, label in enumerate([1]):
            logging.info(
                "Overall Precision, Recall, F1, ROC, PRC, ACC for Class %d: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f",
                label,
                precision_,
                recall_,
                f1_best,
                roc,
                prc,
                acc
            )

        # Append metrics for each repetition
        Precision.append(np.mean(precision_))
        Recall.append(np.mean(recall_))
        ROC.append(roc)
        PRC.append(prc)
        F1_best.append(f1_best)
        ACC.append(acc)

    # Prepare results data
    header = ["Field", "Path", "Method", "Learning Rate", "Precision", "Recall", "F1_best", "F1_best_std", "ROC", "ROC_std", "PRC", "ACC", "all_F1_best"]
    data = [
        args.field, args.path, 'xgb_non_val', args.lr,
        np.mean(Precision), np.mean(Recall), np.mean(F1_best), np.std(F1_best),
        np.mean(ROC), np.std(ROC), np.mean(PRC), np.mean(ACC), F1_best
    ]
    
    # Save results to CSV
    save_results_to_csv("results.csv", header, data)
    logging.info(data)

if __name__ == "__main__":
    main()
