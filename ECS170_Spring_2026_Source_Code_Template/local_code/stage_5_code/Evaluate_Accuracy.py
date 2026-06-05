'''
Concrete Evaluate class for Stage 5 node classification.

Reports accuracy plus weighted and macro precision / recall / F1 so the report
can discuss both overall and per-class-balanced performance.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Evaluate_Accuracy(evaluate):

    def evaluate(self):
        print('evaluating performance...')

        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        accuracy = accuracy_score(y_true, y_pred)
        precision_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_m = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print('Accuracy:', accuracy)
        print('Precision(weighted):', precision_w, '| Precision(macro):', precision_m)
        print('Recall(weighted):', recall_w, '| Recall(macro):', recall_m)
        print('F1(weighted):', f1_w, '| F1(macro):', f1_m)

        result = {
            'accuracy': accuracy,
            'precision_weighted': precision_w,
            'recall_weighted': recall_w,
            'f1_weighted': f1_w,
            'precision_macro': precision_m,
            'recall_macro': recall_m,
            'f1_macro': f1_m,
        }

        for key in ('loss_history', 'train_acc_history', 'val_loss_history', 'val_acc_history'):
            if key in self.data:
                result[key] = self.data[key]
        return result
