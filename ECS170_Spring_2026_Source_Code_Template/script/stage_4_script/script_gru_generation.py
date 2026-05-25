import os
import sys

TEMPLATE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if TEMPLATE_ROOT not in sys.path:
    sys.path.insert(0, TEMPLATE_ROOT)

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN_Generator
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    smoke = os.environ.get('STAGE4_SMOKE') == '1'

    data_obj = Dataset_Loader('joke_generation', '')
    data_obj.task = 'generation'
    data_obj.dataset_source_folder_path = 'data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = 'data'
    data_obj.max_vocab_size = 1000 if smoke else 5000
    data_obj.context_len = 3
    data_obj.max_train_samples = 400 if smoke else None

    method_obj = Method_RNN_Generator('gru text generator', '', cell_type='gru')
    method_obj.plot_filename = 'gru_generation_convergence.png'
    method_obj.max_epoch = 2 if smoke else 60
    method_obj.batch_size = 32 if smoke else 128
    method_obj.embedding_dim = 32 if smoke else 128
    method_obj.hidden_size = 32 if smoke else 128
    method_obj.dropout = 0.2
    method_obj.max_generate_words = 20
    method_obj.temperature = 0.8

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_4_result/GRU_GENERATION_'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = 0

    setting_obj = Setting_Train_Test_Split('train validation split', '')
    evaluate_obj = Evaluate_Accuracy('next_word_accuracy', '')

    print('************ Start Stage 4 GRU Generation ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    scores, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance (GRU generation validation) ************')
    print('Next-word Accuracy:', scores['accuracy'])
    print('Generated Samples:', scores.get('generated_samples'))
    print('************ Finish ************')
