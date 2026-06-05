'''
Concrete SettingModule class for Stage 5 transductive node classification.

The dataset loader already produces the full graph together with the
class-balanced train / validation / test index split, so this setting simply
hands that bundle to the GCN method, saves the raw predictions, and evaluates.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):

    def run_and_return(self):
        '''Load the graph, train+test the method, and return the raw result and
        the balanced validation score (used for seed / model selection).  Does
        NOT save or evaluate, so it can be called repeatedly across seeds.'''
        loaded_data = self.dataset.load()
        self.method.data = {
            'graph': loaded_data['graph'],
            'train_test_val': loaded_data['train_test_val'],
        }
        learned_result = self.method.run()
        return learned_result, self.method.best_val_score

    def load_run_save_evaluate(self):
        learned_result, _ = self.run_and_return()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None
