'''
Concrete SettingModule class for Stage 4 train/test experimental setting.
'''

from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):

        loaded_data = self.dataset.load()

        X_train = loaded_data['train_X']
        y_train = loaded_data['train_y']
        X_test = loaded_data['test_X']
        y_test = loaded_data['test_y']

        self.method.data = {
            'task': loaded_data.get('task'),
            'train': {
                'X': X_train,
                'y': y_train,
                'lengths': loaded_data.get('train_lengths'),
            },
            'test': {
                'X': X_test,
                'y': y_test,
                'lengths': loaded_data.get('test_lengths'),
            },
            'word_to_id': loaded_data.get('word_to_id'),
            'id_to_word': loaded_data.get('id_to_word'),
            'vocab_size': loaded_data.get('vocab_size'),
            'context_len': loaded_data.get('context_len'),
            'seed_prompts': loaded_data.get('seed_prompts'),
            'jokes': loaded_data.get('jokes'),
        }

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None
