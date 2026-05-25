from local_code.base_class.method import method

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def _pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _build_recurrent(cell_type, input_size, hidden_size, num_layers, dropout, bidirectional):
    cell_type = cell_type.lower()
    dropout_value = dropout if num_layers > 1 else 0.0
    if cell_type == 'lstm':
        return nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_value,
            bidirectional=bidirectional,
        )
    if cell_type == 'gru':
        return nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_value,
            bidirectional=bidirectional,
        )
    if cell_type == 'rnn':
        return nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout_value,
            bidirectional=bidirectional,
        )
    raise ValueError('cell_type must be rnn, lstm, or gru')


class Method_RNN_Classifier(method, nn.Module):
    data = None
    max_epoch = 8
    learning_rate = 1e-3
    batch_size = 128
    weight_decay = 1e-4
    embedding_dim = 128
    hidden_size = 128
    num_layers = 1
    dropout = 0.3
    bidirectional = True
    num_workers = 0
    validation_ratio = 0.1
    gradient_clip = 1.0
    plot_filename = 'rnn_classification_convergence.png'

    def __init__(
            self,
            mName,
            mDescription,
            vocab_size=None,
            cell_type='rnn',
            num_classes=2,
            embedding_dim=None,
            hidden_size=None,
            num_layers=None,
            dropout=None,
            bidirectional=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.cell_type = cell_type.lower()
        self.num_classes = num_classes
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if num_layers is not None:
            self.num_layers = num_layers
        if dropout is not None:
            self.dropout = dropout
        if bidirectional is not None:
            self.bidirectional = bidirectional
        self.embedding = None
        self.rnn = None
        self.dropout_layer = None
        self.fc = None
        if vocab_size is not None:
            self._build_network(vocab_size)
        self.loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []

    def _build_network(self, vocab_size):
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.rnn = _build_recurrent(
            self.cell_type,
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.bidirectional,
        )
        out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(out_dim, self.num_classes)

    def forward(self, x, lengths=None):
        if self.embedding is None:
            raise RuntimeError('network has not been built; vocab_size is missing')
        emb = self.embedding(x)
        output, _ = self.rnn(emb)

        if lengths is None:
            final_output = output[:, -1, :]
        else:
            lengths = lengths.to(x.device).clamp(min=1, max=x.size(1))
            time_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.size(2))
            final_output = output.gather(1, time_index).squeeze(1)
        final_output = self.dropout_layer(final_output)
        return self.fc(final_output)

    def _make_loaders(self, X, y, lengths):
        n = len(y)
        rng = np.random.default_rng(2)
        indices = rng.permutation(n)
        val_size = int(n * self.validation_ratio)
        val_size = max(1, val_size)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_tensor = torch.as_tensor(X, dtype=torch.long)
        y_tensor = torch.as_tensor(y, dtype=torch.long)
        length_tensor = torch.as_tensor(lengths, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(
            X_tensor[train_idx],
            y_tensor[train_idx],
            length_tensor[train_idx],
        )
        val_dataset = torch.utils.data.TensorDataset(
            X_tensor[val_idx],
            y_tensor[val_idx],
            length_tensor[val_idx],
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return train_loader, val_loader

    def fit(self, X, y, lengths):
        if self.embedding is None:
            self._build_network(self.data['vocab_size'])
        device = _pick_device()
        print('Using device:', device)
        print('Cell type:', self.cell_type, '| bidirectional:', self.bidirectional)
        self.to(device)

        train_loader, val_loader = self._make_loaders(X, y, lengths)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        best_state = None
        best_val_acc = -1.0

        for epoch in range(self.max_epoch):
            self.train(mode=True)
            running_loss = 0.0
            train_correct = 0
            train_total = 0

            for xb, yb, lb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                lb = lb.to(device)

                optimizer.zero_grad()
                logits = self.forward(xb, lb)
                loss = loss_fn(logits, yb)
                loss.backward()
                if self.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
                optimizer.step()

                running_loss += loss.item() * xb.size(0)
                train_correct += (logits.argmax(dim=1) == yb).sum().item()
                train_total += xb.size(0)

            avg_loss = running_loss / train_total
            train_acc = train_correct / train_total
            val_acc = self._evaluate_loader(val_loader, device)
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(train_acc)
            self.val_accuracy_history.append(val_acc)

            print('Epoch:', epoch, 'Loss:', avg_loss, 'Train Accuracy:', train_acc, 'Val Accuracy:', val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}

        if best_state is not None:
            self.load_state_dict(best_state)
        self._save_plot('Text Classification Training Convergence')

    def _evaluate_loader(self, loader, device):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb, lb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                lb = lb.to(device)
                logits = self.forward(xb, lb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += xb.size(0)
        return correct / total

    def _save_plot(self, title):
        out_dir = 'result/stage_4_result'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, self.plot_filename)
        plt.figure()
        plt.plot(range(len(self.loss_history)), self.loss_history, label='Training Loss')
        plt.plot(range(len(self.accuracy_history)), self.accuracy_history, label='Training Accuracy')
        plt.plot(range(len(self.val_accuracy_history)), self.val_accuracy_history, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.savefig(out_path)
        plt.close()

    def test(self, X, lengths):
        device = _pick_device()
        self.to(device)
        self.eval()

        X_tensor = torch.as_tensor(X, dtype=torch.long)
        length_tensor = torch.as_tensor(lengths, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_tensor, length_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        pred_chunks = []
        with torch.no_grad():
            for xb, lb in loader:
                xb = xb.to(device)
                lb = lb.to(device)
                logits = self.forward(xb, lb)
                pred_chunks.append(logits.argmax(dim=1).detach().cpu())
        return torch.cat(pred_chunks).numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(
            self.data['train']['X'],
            self.data['train']['y'],
            self.data['train']['lengths'],
        )

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'], self.data['test']['lengths'])
        true_y = np.asarray(self.data['test']['y'], dtype=np.int64)

        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history,
            'val_accuracy_history': self.val_accuracy_history,
            'cell_type': self.cell_type,
        }


class Method_RNN_Generator(method, nn.Module):
    data = None
    max_epoch = 40
    learning_rate = 1e-3
    batch_size = 128
    weight_decay = 1e-5
    embedding_dim = 128
    hidden_size = 128
    num_layers = 1
    dropout = 0.2
    bidirectional = False
    num_workers = 0
    gradient_clip = 1.0
    plot_filename = 'rnn_generation_convergence.png'
    max_generate_words = 25
    temperature = 0.8

    def __init__(
            self,
            mName,
            mDescription,
            vocab_size=None,
            cell_type='rnn',
            embedding_dim=None,
            hidden_size=None,
            num_layers=None,
            dropout=None,
            bidirectional=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.cell_type = cell_type.lower()
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if num_layers is not None:
            self.num_layers = num_layers
        if dropout is not None:
            self.dropout = dropout
        if bidirectional is not None:
            self.bidirectional = bidirectional
        self.embedding = None
        self.rnn = None
        self.dropout_layer = None
        self.fc = None
        if vocab_size is not None:
            self._build_network(vocab_size)
        self.loss_history = []
        self.val_accuracy_history = []

    def _build_network(self, vocab_size):
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.rnn = _build_recurrent(
            self.cell_type,
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.bidirectional,
        )
        out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(out_dim, vocab_size)

    def forward(self, x):
        if self.embedding is None:
            raise RuntimeError('network has not been built; vocab_size is missing')
        emb = self.embedding(x)
        output, _ = self.rnn(emb)
        last_output = self.dropout_layer(output[:, -1, :])
        return self.fc(last_output)

    def fit(self, X, y, X_val, y_val):
        if self.embedding is None:
            self._build_network(self.data['vocab_size'])
        device = _pick_device()
        print('Using device:', device)
        print('Cell type:', self.cell_type)
        self.to(device)

        train_dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(X, dtype=torch.long),
            torch.as_tensor(y, dtype=torch.long),
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(X_val, dtype=torch.long),
            torch.as_tensor(y_val, dtype=torch.long),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        best_state = None
        best_val_acc = -1.0

        for epoch in range(self.max_epoch):
            self.train(mode=True)
            running_loss = 0.0
            train_total = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                if self.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
                optimizer.step()

                running_loss += loss.item() * xb.size(0)
                train_total += xb.size(0)

            avg_loss = running_loss / train_total
            val_acc = self._evaluate_loader(val_loader, device)
            self.loss_history.append(avg_loss)
            self.val_accuracy_history.append(val_acc)
            print('Epoch:', epoch, 'Loss:', avg_loss, 'Val Next-word Accuracy:', val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}

        if best_state is not None:
            self.load_state_dict(best_state)
        self._save_plot('Text Generation Training Convergence')

    def _evaluate_loader(self, loader, device):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = self.forward(xb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += xb.size(0)
        return correct / total

    def _save_plot(self, title):
        out_dir = 'result/stage_4_result'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, self.plot_filename)
        plt.figure()
        plt.plot(range(len(self.loss_history)), self.loss_history, label='Training Loss')
        plt.plot(range(len(self.val_accuracy_history)), self.val_accuracy_history, label='Validation Next-word Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.savefig(out_path)
        plt.close()

    def test(self, X):
        device = _pick_device()
        self.to(device)
        self.eval()
        dataset = torch.utils.data.TensorDataset(torch.as_tensor(X, dtype=torch.long))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        pred_chunks = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                logits = self.forward(xb)
                pred_chunks.append(logits.argmax(dim=1).detach().cpu())
        return torch.cat(pred_chunks).numpy()

    def generate(self, seed_text, max_words=None, temperature=None):
        if max_words is None:
            max_words = self.max_generate_words
        if temperature is None:
            temperature = self.temperature

        word_to_id = self.data['word_to_id']
        id_to_word = self.data['id_to_word']
        context_len = self.data['context_len']
        unk_id = word_to_id.get('<UNK>', 1)
        eos_id = word_to_id.get('<EOS>', 2)

        words = seed_text.lower().split()
        if len(words) < context_len:
            raise ValueError('seed_text must contain at least context_len words')
        generated = list(words)

        device = _pick_device()
        self.to(device)
        self.eval()

        with torch.no_grad():
            for _ in range(max_words):
                context_words = generated[-context_len:]
                context_ids = [word_to_id.get(word, unk_id) for word in context_words]
                x = torch.as_tensor([context_ids], dtype=torch.long).to(device)
                logits = self.forward(x).squeeze(0)
                if temperature <= 0:
                    next_id = int(torch.argmax(logits).item())
                else:
                    probs = torch.softmax(logits / temperature, dim=0)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
                if next_id == eos_id:
                    break
                next_word = id_to_word.get(next_id, '<UNK>')
                if next_word in ('<PAD>', '<UNK>'):
                    continue
                generated.append(next_word)

        return ' '.join(generated)

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(
            self.data['train']['X'],
            self.data['train']['y'],
            self.data['test']['X'],
            self.data['test']['y'],
        )

        print('--start validation testing...')
        pred_y = self.test(self.data['test']['X'])
        true_y = np.asarray(self.data['test']['y'], dtype=np.int64)

        generated_samples = {}
        for prompt in self.data.get('seed_prompts') or []:
            generated_samples[prompt] = self.generate(prompt)
        generated_samples['the movie was'] = self.generate('the movie was')

        print('--generated samples--')
        for prompt, text in generated_samples.items():
            print(prompt + ':', text)

        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'loss_history': self.loss_history,
            'val_accuracy_history': self.val_accuracy_history,
            'generated_samples': generated_samples,
            'cell_type': self.cell_type,
        }
