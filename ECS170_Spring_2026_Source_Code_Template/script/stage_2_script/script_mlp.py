from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('mnist_digits', '')

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy_precision_recall_f1', '')

    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    scores, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    print('MLP Accuracy:', scores['accuracy'])
    print('MLP Precision(weighted):', scores['precision_weighted'])
    print('MLP Recall(weighted):', scores['recall_weighted'])
    print('MLP F1(weighted):', scores['f1_weighted'])
    print('************ Finish ************')