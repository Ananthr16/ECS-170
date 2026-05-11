from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- CNN on ORL faces (Stage 3) ----
if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('orl_faces', '')
    data_obj.dataset_source_folder_path = 'data/stage_3_data-2/'
    data_obj.dataset_source_file_name = 'ORL'

    method_obj = Method_CNN('convolutional neural network', '', num_classes=40, in_channels=1)
    method_obj.plot_filename = 'cnn_orl_convergence.png'
    method_obj.max_epoch = 40
    method_obj.batch_size = 32

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_3_result/CNN_ORL_'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = 0

    setting_obj = Setting_Train_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy_precision_recall_f1', '')

    print('************ Start ORL ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    scores, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance (ORL test) ************')
    print('Accuracy:', scores['accuracy'])
    print('Precision(weighted):', scores['precision_weighted'])
    print('Recall(weighted):', scores['recall_weighted'])
    print('F1(weighted):', scores['f1_weighted'])
    print('************ Finish ************')
