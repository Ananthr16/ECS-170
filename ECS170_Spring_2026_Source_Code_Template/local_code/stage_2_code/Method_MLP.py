from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 100
    learning_rate = 1e-3
    batch_size = 256
    weight_decay = 1e-4
    dropout_p = 0.1
    label_smoothing = 0.05

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Deeper MLP: Linear → BatchNorm → ReLU → Dropout (last layer: logits only)
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _pick_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def fit(self, X, y):
        device = self._pick_device()
        print('Using device:', device)
        self.to(device)

        X_tensor = torch.FloatTensor(np.asarray(X, dtype=np.float32))
        y_tensor = torch.LongTensor(np.asarray(y, dtype=np.int64))

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epoch
        )
        try:
            loss_function = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        except TypeError:
            loss_function = nn.CrossEntropyLoss()

        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        loss_history = []
        accuracy_history = []

        for epoch in range(self.max_epoch):
            self.train(True)
            running_loss = 0.0
            pred_chunks = []
            label_chunks = []

            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = self.forward(xb)
                batch_loss = loss_function(logits, yb)
                batch_loss.backward()
                optimizer.step()

                running_loss += batch_loss.item() * xb.size(0)
                pred_chunks.append(logits.argmax(dim=1).detach().cpu())
                label_chunks.append(yb.detach().cpu())

            scheduler.step()

            all_pred = torch.cat(pred_chunks).numpy()
            all_true = torch.cat(label_chunks).numpy()
            avg_loss = running_loss / len(dataset)
            train_acc = float((all_pred == all_true).mean())
            loss_history.append(avg_loss)
            accuracy_history.append(train_acc)

            print('Epoch:', epoch, 'Accuracy:', train_acc, 'Loss:', avg_loss)
            if epoch % 10 == 0 or epoch == self.max_epoch - 1:
                accuracy_evaluator.data = {'true_y': all_true, 'pred_y': all_pred}
                accuracy_evaluator.evaluate()

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
        device = self._pick_device()
        self.to(device)
        self.eval()

        X_tensor = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)

        with torch.no_grad():
            y_pred = self.forward(X_tensor)

        return y_pred.argmax(dim=1).detach().cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y']
        }
