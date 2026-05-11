from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- CNN on CIFAR-10 colored objects (Stage 3) ----
# Strong recipe: ResNet-18 adapted for 32x32, train-time augmentation, SGD + cosine.
# Expect long runtime (many epochs). ~90%+ is typical; reaching ~95% may need 200+ epochs
# or further tuning (autoaug, wider network, etc.).
if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('cifar10_objects', '')
    data_obj.dataset_source_folder_path = 'data/stage_3_data-2/'
    data_obj.dataset_source_file_name = 'CIFAR'

    method_obj = Method_CNN(
        'convolutional neural network',
        '',
        num_classes=10,
        in_channels=3,
        backbone='resnet18_cifar',
    )
    method_obj.plot_filename = 'cnn_cifar_convergence.png'
    method_obj.max_epoch = 200
    method_obj.batch_size = 256
    method_obj.use_autocast = True
    method_obj.augment_train = True
    method_obj.optimizer_type = 'sgd'
    method_obj.learning_rate = 0.1
    method_obj.weight_decay = 5e-4
    method_obj.label_smoothing = 0.05
    method_obj.sgd_momentum = 0.9

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_3_result/CNN_CIFAR_'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = 0

    setting_obj = Setting_Train_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy_precision_recall_f1', '')

    print('************ Start CIFAR ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    scores, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance (CIFAR test) ************')
    print('Accuracy:', scores['accuracy'])
    print('Precision(weighted):', scores['precision_weighted'])
    print('Recall(weighted):', scores['recall_weighted'])
    print('F1(weighted):', scores['f1_weighted'])
    print('************ Finish ************')
