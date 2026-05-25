import os
import sys

TEMPLATE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if TEMPLATE_ROOT not in sys.path:
    sys.path.insert(0, TEMPLATE_ROOT)

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN_Classifier
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    smoke = os.environ.get('STAGE4_SMOKE') == '1'

    data_obj = Dataset_Loader('imdb_sentiment', '')
    data_obj.task = 'classification'
    data_obj.dataset_source_folder_path = 'data/stage_4_data/text_classification/'
    data_obj.max_vocab_size = 1000 if smoke else 20000
    data_obj.max_len = 80 if smoke else 400
    data_obj.max_train_samples = 200 if smoke else None
    data_obj.max_test_samples = 100 if smoke else None

    method_obj = Method_RNN_Classifier('rnn text classifier', '', cell_type='rnn')
    method_obj.plot_filename = 'rnn_classification_convergence.png'
    method_obj.max_epoch = 1 if smoke else 8
    method_obj.batch_size = 32 if smoke else 128
    method_obj.embedding_dim = 32 if smoke else 128
    method_obj.hidden_size = 32 if smoke else 128
    method_obj.dropout = 0.2 if smoke else 0.3
    method_obj.bidirectional = True

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_4_result/RNN_CLASSIFICATION_'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = 0

    setting_obj = Setting_Train_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy_precision_recall_f1', '')

    print('************ Start Stage 4 RNN Classification ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    scores, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance (RNN classification test) ************')
    print('Accuracy:', scores['accuracy'])
    print('Precision(weighted):', scores['precision_weighted'])
    print('Recall(weighted):', scores['recall_weighted'])
    print('F1(weighted):', scores['f1_weighted'])
    print('************ Finish ************')
