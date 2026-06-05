'''
Concrete MethodModule class for Stage 5: a standard Graph Convolutional Network
(GCN) for transductive node classification, following Kipf & Welling
(ICLR 2017, "Semi-Supervised Classification with Graph Convolutional Networks").

The network is the canonical two-layer GCN.  Each graph-convolution layer
propagates information over the symmetrically-normalized adjacency
        A_hat = D^-1/2 (A + I) D^-1/2
via
        H^(l+1) = sigma( A_hat * H^(l) * W^(l) ),
so the full model is
        H1 = ReLU( A_hat * X  * W0 )
        Y  =        A_hat * H1 * W1   (logits).

Training is full-batch and transductive: every node's features and the whole
graph are visible during the forward pass, but only the labels of the *training*
nodes contribute to the loss.  A held-out, class-balanced validation set is used
for early stopping and best-model selection (on balanced validation accuracy,
which matches the class-balanced test set defined in the Stage 5 ReadMe).
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
# Extended for ECS 170 Stage 5 (GCN node classification).

from local_code.base_class.method import method
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _pick_device():
    """Prefer CUDA, then Apple-Silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _balanced_accuracy(pred, true, num_classes):
    """Mean per-class recall (a.k.a. balanced accuracy).

    Matches the class-balanced test set far better than plain accuracy or loss
    when the validation distribution is not perfectly uniform, so it is a
    faithful, low-variance signal for early stopping and seed selection.
    """
    recalls = []
    for c in range(num_classes):
        mask = (true == c)
        denom = int(mask.sum().item())
        if denom == 0:
            continue
        recalls.append((pred[mask] == c).float().mean().item())
    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


class GraphConvolution(nn.Module):
    '''A single spectral graph-convolution layer:  out = A_hat * (X * W) + b.'''

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot / Xavier uniform initialization, as in the original GCN paper.
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        if adj.is_sparse:
            output = torch.sparse.mm(adj, support)
        else:
            output = torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class Method_GCN(method, nn.Module):
    data = None

    # ---- default hyper-parameters (overridable from the run scripts) ----
    max_epoch = 400
    learning_rate = 1e-2
    weight_decay = 5e-4      # L2 applied to the first graph-convolution layer
    hidden_dim = 64
    dropout = 0.5
    patience = 100           # early-stopping patience (epochs without val improvement)
    plot_filename = 'gcn_convergence.png'
    result_folder_path = 'result/stage_5_result'
    best_val_score = None    # best balanced validation accuracy reached during fit
    save_plots = True        # set False during multi-seed selection, then plot the winner

    def __init__(self, mName, mDescription,
                 in_dim=None, num_classes=None,
                 hidden_dim=None, dropout=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.in_dim = in_dim
        self.num_classes = num_classes
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        if dropout is not None:
            self.dropout = dropout

        # built lazily once the data dimensionality is known (see fit())
        self.gc1 = None
        self.gc2 = None
        if in_dim is not None and num_classes is not None:
            self._build_network(in_dim, num_classes)

        self.loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def _build_network(self, in_dim, num_classes):
        self.in_dim = in_dim
        self.num_classes = num_classes
        # Canonical two-layer GCN: input -> hidden (ReLU) -> classes.
        self.gc1 = GraphConvolution(in_dim, self.hidden_dim)
        self.gc2 = GraphConvolution(self.hidden_dim, num_classes)

    def forward(self, x, adj):
        '''Forward propagation returning class logits for every node.

        Dropout is applied to the input of each graph-convolution layer, exactly
        as in the original GCN implementation (Kipf & Welling, 2017).
        '''
        x = F.dropout(x, self.dropout, training=self.training)
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.gc2(h, adj)

    def _resolve_device(self, adj):
        '''Pick a device and make sure sparse matmul actually works on it.

        Apple MPS does not always support sparse ops; if a probe fails we fall
        back to CPU so training still runs (just slower) on any machine.
        '''
        device = _pick_device()
        if device.type == 'mps' and adj.is_sparse:
            try:
                probe = torch.sparse.mm(
                    adj.to(device),
                    torch.ones(adj.shape[1], 1, device=device))
                _ = probe.sum().item()
            except Exception:
                print('  [device] sparse ops unsupported on MPS -> falling back to CPU')
                device = torch.device('cpu')
        return device

    def fit(self, X, adj, y, idx_train, idx_val):
        if self.gc1 is None:
            self._build_network(X.shape[1], int(y.max().item()) + 1)

        device = self._resolve_device(adj)
        print('Using device:', device)
        print('Architecture: GCN({} -> {} -> {}) | dropout={} | wd={} | lr={}'.format(
            self.in_dim, self.hidden_dim, self.num_classes,
            self.dropout, self.weight_decay, self.learning_rate))

        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.to(device)
        X = X.to(device)
        adj = adj.to(device)
        y = y.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)

        # Weight decay (L2) is applied only to the first graph-convolution layer,
        # following the original GCN setup; the output layer is left unregularized.
        optimizer = torch.optim.Adam([
            {'params': self.gc1.parameters(), 'weight_decay': self.weight_decay},
            {'params': self.gc2.parameters(), 'weight_decay': 0.0},
        ], lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # Model selection: keep the weights with the best *balanced* validation
        # accuracy (mean per-class recall) and early-stop when it stops improving.
        best_val_score = -1.0
        best_val_acc = 0.0
        best_state = None
        epochs_no_improve = 0
        num_classes = int(y.max().item()) + 1

        for epoch in range(self.max_epoch):
            # ---- training step (full-batch) ----
            self.train(mode=True)
            optimizer.zero_grad()
            logits = self.forward(X, adj)
            train_loss = loss_fn(logits[idx_train], y[idx_train])
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_pred = logits[idx_train].argmax(dim=1)
                train_acc = (train_pred == y[idx_train]).float().mean().item()

            # ---- validation step (no dropout) ----
            self.eval()
            with torch.no_grad():
                val_logits = self.forward(X, adj)
                val_loss = loss_fn(val_logits[idx_val], y[idx_val]).item()
                val_pred = val_logits[idx_val].argmax(dim=1)
                val_acc = (val_pred == y[idx_val]).float().mean().item()
                val_bal_acc = _balanced_accuracy(val_pred, y[idx_val], num_classes)

            self.loss_history.append(train_loss.item())
            self.train_acc_history.append(train_acc)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)

            if epoch % 20 == 0 or epoch == self.max_epoch - 1:
                print('Epoch: {:4d} | Train Loss: {:.4f} Acc: {:.4f} | '
                      'Val Loss: {:.4f} Acc: {:.4f} BalAcc: {:.4f}'.format(
                          epoch, train_loss.item(), train_acc,
                          val_loss, val_acc, val_bal_acc))

            # ---- early stopping on balanced validation accuracy (keep best) ----
            if val_bal_acc > best_val_score + 1e-4:
                best_val_score = val_bal_acc
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print('Early stopping at epoch {} (best balanced val acc: '
                          '{:.4f})'.format(epoch, best_val_score))
                    break

        self.best_val_score = best_val_score
        if best_state is not None:
            self.load_state_dict(best_state)
        print('Best balanced validation accuracy: {:.4f} | validation accuracy: '
              '{:.4f}'.format(best_val_score, best_val_acc))
        if self.save_plots:
            self._save_learning_curves()

    def _save_learning_curves(self):
        os.makedirs(self.result_folder_path, exist_ok=True)
        out_path = os.path.join(self.result_folder_path, self.plot_filename)
        epochs = range(len(self.loss_history))

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        ax_loss.plot(epochs, self.loss_history, label='Training Loss')
        ax_loss.plot(epochs, self.val_loss_history, label='Validation Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('GCN Loss Curve')
        ax_loss.legend()

        ax_acc.plot(epochs, self.train_acc_history, label='Training Accuracy')
        ax_acc.plot(epochs, self.val_acc_history, label='Validation Accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title('GCN Accuracy Curve')
        ax_acc.legend()

        fig.suptitle(self.method_name)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print('Saved learning curves to', out_path)

    def test(self, X, adj, idx_test):
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            logits = self.forward(X.to(device), adj.to(device))
            pred = logits[idx_test.to(device)].argmax(dim=1)
        return pred.detach().cpu().numpy()

    def run(self):
        print('method running...')
        graph = self.data['graph']
        ttv = self.data['train_test_val']

        X = graph['X']
        adj = graph['utility']['A']
        y = graph['y']
        idx_train = ttv['idx_train']
        idx_val = ttv['idx_val']
        idx_test = ttv['idx_test']

        print('--start training...')
        self.fit(X, adj, y, idx_train, idx_val)

        print('--start testing...')
        pred_y = self.test(X, adj, idx_test)
        true_y = y[idx_test].detach().cpu().numpy()

        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_loss_history': self.val_loss_history,
            'val_acc_history': self.val_acc_history,
        }
