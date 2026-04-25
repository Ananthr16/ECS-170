from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):

    def evaluate(self):
        print('evaluating performance...')

        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print('Accuracy:', accuracy)
        print('Precision(weighted):', precision)
        print('Recall(weighted):', recall)
        print('F1(weighted):', f1)

        return {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1
        }