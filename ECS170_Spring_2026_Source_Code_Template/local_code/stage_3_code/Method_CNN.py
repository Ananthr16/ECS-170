from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn

try:
    from torchvision.models import resnet18
except ImportError as _e:
    resnet18 = None

try:
    from torchvision.transforms import v2 as _tv2
except ImportError:
    _tv2 = None


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 15
    learning_rate = 1e-3
    batch_size = 128
    weight_decay = 1e-4
    label_smoothing = 0.05
    plot_filename = 'cnn_convergence.png'
    backbone = 'custom'
    augment_train = False
    optimizer_type = 'adamw'
    sgd_momentum = 0.9
    num_workers = 0
    use_autocast = False

    def __init__(self, mName, mDescription, num_classes=10, in_channels=1, backbone=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.in_channels = in_channels
        if backbone is not None:
            self.backbone = backbone

        if self.backbone == 'resnet18_cifar':
            if resnet18 is None:
                raise ImportError('torchvision is required for backbone resnet18_cifar')
            self.net = self._build_resnet18_cifar(num_classes)
        else:
            c = in_channels
            self.net = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, num_classes),
            )

        self._train_aug = None

    @staticmethod
    def _build_resnet18_cifar(num_classes):
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def _pick_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _maybe_augment(self, xb):
        aug = self._train_aug
        if aug is None or not self.training:
            return xb
        return aug(xb)

    def fit(self, X, y):
        device = self._pick_device()
        if self.augment_train and _tv2 is not None:
            self._train_aug = _tv2.Compose([
                _tv2.RandomCrop(32, padding=4),
                _tv2.RandomHorizontalFlip(),
            ])
        else:
            self._train_aug = None

        print('Using device:', device)
        print('Backbone:', self.backbone, '| augment_train:', self.augment_train and self._train_aug is not None)
        print('batch_size:', self.batch_size, '| use_autocast:', self.use_autocast)

        if device.type in ('mps', 'cuda'):
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.to(device)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_arr),
            torch.from_numpy(y_arr),
        )
        loader_kw = dict(
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
        )
        if self.num_workers > 0:
            loader_kw['persistent_workers'] = True
            loader_kw['prefetch_factor'] = 2
        loader = torch.utils.data.DataLoader(dataset, **loader_kw)

        use_amp = self.use_autocast and device.type in ('cuda', 'mps')

        if self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=self.sgd_momentum,
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epoch
        )

        try:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        except TypeError:
            loss_fn = nn.CrossEntropyLoss()

        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        loss_history = []
        accuracy_history = []

        for epoch in range(self.max_epoch):
            self.train(mode=True)
            running_loss = 0.0
            pred_chunks = []
            label_chunks = []

            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                xb = self._maybe_augment(xb)

                optimizer.zero_grad()
                if use_amp:
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        logits = self.forward(xb)
                        loss = loss_fn(logits, yb)
                else:
                    logits = self.forward(xb)
                    loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)
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
            if epoch % 5 == 0 or epoch == self.max_epoch - 1:
                accuracy_evaluator.data = {'true_y': all_true, 'pred_y': all_pred}
                accuracy_evaluator.evaluate()

        out_dir = 'result/stage_3_result'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, self.plot_filename)

        plt.figure()
        plt.plot(range(self.max_epoch), loss_history, label='Training Loss')
        plt.plot(range(self.max_epoch), accuracy_history, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('CNN Training Convergence')
        plt.legend()
        plt.savefig(out_path)
        plt.close()

    def test(self, X):
        device = self._pick_device()
        self.to(device)
        self.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X_arr).to(device)

        with torch.no_grad():
            logits = self.forward(X_tensor)
            pred = logits.argmax(dim=1).detach().cpu().numpy()

        return pred

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        true_y = np.asarray(self.data['test']['y'], dtype=np.int64)

        return {
            'pred_y': pred_y,
            'true_y': true_y,
        }
