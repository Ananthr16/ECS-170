from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 1250
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.fc_layer_1 = nn.Linear(784, 128)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(128, 64)
        self.activation_func_2 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(64, 10)

    def forward(self, x):
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        y_pred = self.fc_layer_3(h2)
        return y_pred

    def train(self, X, y):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(device)

        X_tensor = torch.FloatTensor(np.array(X)).to(device)
        y_tensor = torch.LongTensor(np.array(y)).to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        loss_history = []
        accuracy_history = []

        for epoch in range(self.max_epoch):
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_tensor)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            pred_labels = y_pred.max(1)[1].detach().cpu()
            true_labels = y_tensor.detach().cpu()

            accuracy_evaluator.data = {
                'true_y': true_labels,
                'pred_y': pred_labels
            }

            scores = accuracy_evaluator.evaluate()

            loss_history.append(train_loss.item())
            accuracy_history.append(scores['accuracy'])

            print(
                'Epoch:', epoch,
                'Accuracy:', scores['accuracy'],
                'Loss:', train_loss.item()
            )

        plt.figure()
        plt.plot(range(self.max_epoch), loss_history, label='Training Loss')
        plt.plot(range(self.max_epoch), accuracy_history, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('MLP Training Convergence')
        plt.legend()
        plt.savefig('result/stage_2_result/mlp_convergence_plot.png')
        plt.close()

    def test(self, X):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(device)

        X_tensor = torch.FloatTensor(np.array(X)).to(device)

        with torch.no_grad():
            y_pred = self.forward(X_tensor)

        return y_pred.max(1)[1].detach().cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y']
        }