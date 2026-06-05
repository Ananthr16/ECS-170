'''
Stage 5 runner: GCN node classification on the Pubmed citation network.

Trains the GCN under several random seeds and selects the run with the best
balanced validation accuracy (model selection never uses the test labels).

Run from the project root:  python script/stage_5_script/script_gcn_pubmed.py
'''

import os
import sys

# make the project root importable when this script is launched directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch

from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_GCN import Method_GCN
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy


DATASET_NAME = 'pubmed'
DATASET_PATH = 'data/stage_5_data/pubmed'
PLOT_FILENAME = 'gcn_pubmed_convergence.png'
RESULT_PREFIX = 'result/stage_5_result/GCN_PUBMED_'
CANDIDATE_SEEDS = [1, 2, 3, 4, 5]


def build_method():
    method_obj = Method_GCN('GCN (Pubmed)', '', hidden_dim=64, dropout=0.5)
    method_obj.learning_rate = 1e-2
    method_obj.weight_decay = 5e-4
    method_obj.max_epoch = 400
    method_obj.patience = 100
    method_obj.plot_filename = PLOT_FILENAME
    return method_obj


def main(seeds=None):
    seeds = seeds if seeds is not None else CANDIDATE_SEEDS
    evaluate_obj = Evaluate_Accuracy('accuracy_precision_recall_f1', '')

    print('************ Start Pubmed ************')
    best = None
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        data_obj = Dataset_Loader(seed=seed, dName=DATASET_NAME, dDescription='Pubmed citation network')
        data_obj.dataset_name = DATASET_NAME
        data_obj.dataset_source_folder_path = DATASET_PATH

        method_obj = build_method()
        method_obj.save_plots = False

        setting_obj = Setting_Train_Test_Split('train test split', '')
        setting_obj.prepare(data_obj, method_obj, None, evaluate_obj)
        result, val_score = setting_obj.run_and_return()
        print('---- seed {}: balanced validation accuracy = {:.4f} ----'.format(seed, val_score))

        if best is None or val_score > best['val_score']:
            best = {'val_score': val_score, 'seed': seed, 'result': result, 'method': method_obj}

    print('************ Selected seed {} (balanced val acc {:.4f}) ************'.format(
        best['seed'], best['val_score']))

    best['method'].save_plots = True
    best['method']._save_learning_curves()

    best['result']['selected_seed'] = best['seed']
    best['result']['best_val_score'] = best['val_score']

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_PREFIX
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = 0
    result_obj.data = best['result']
    result_obj.save()

    evaluate_obj.data = best['result']
    scores = evaluate_obj.evaluate()

    print('************ Overall Performance (Pubmed test) ************')
    print('Accuracy:', scores['accuracy'])
    print('Precision(weighted):', scores['precision_weighted'])
    print('Recall(weighted):', scores['recall_weighted'])
    print('F1(weighted):', scores['f1_weighted'])
    print('************ Finish ************')
    return scores


if __name__ == '__main__':
    main()
