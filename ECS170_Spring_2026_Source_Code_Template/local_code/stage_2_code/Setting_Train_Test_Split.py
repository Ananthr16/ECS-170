'''
Concrete SettingModule class for Stage 2 train/test experimental setting
'''

from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self.dataset.load()

        X_train = loaded_data['train_X']
        y_train = loaded_data['train_y']
        X_test = loaded_data['test_X']
        y_test = loaded_data['test_y']

        # run MethodModule
        self.method.data = {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }

        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        # evaluate performance
        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None