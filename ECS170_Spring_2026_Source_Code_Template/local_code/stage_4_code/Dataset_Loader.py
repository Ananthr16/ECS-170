'''
Concrete IO class for Stage 4 text classification and text generation datasets.
'''

import csv
import html
import os
import random
import re
from collections import Counter

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    task = 'classification'
    max_vocab_size = 20000
    min_freq = 2
    max_len = 400
    context_len = 3
    validation_ratio = 0.1
    random_seed = 2
    max_train_samples = None
    max_test_samples = None

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    EOS_TOKEN = '<EOS>'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        if self.task == 'generation':
            return self._load_generation()
        if self.task == 'classification':
            return self._load_classification()
        raise ValueError('task must be either classification or generation')

    def _resolve_path(self):
        folder = self.dataset_source_folder_path or ''
        file_name = self.dataset_source_file_name or ''
        if file_name:
            path = os.path.join(folder, file_name)
        else:
            path = folder
        return os.path.normpath(path)

    @staticmethod
    def _clean_text(text):
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = text.lower()
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)

    @staticmethod
    def _build_vocab(token_lists, max_vocab_size, min_freq, extra_tokens):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        word_to_id = {token: idx for idx, token in enumerate(extra_tokens)}
        for word, count in counter.most_common():
            if count < min_freq:
                continue
            if word in word_to_id:
                continue
            if len(word_to_id) >= max_vocab_size:
                break
            word_to_id[word] = len(word_to_id)

        id_to_word = {idx: word for word, idx in word_to_id.items()}
        return word_to_id, id_to_word, counter

    def _encode_and_pad(self, tokens, word_to_id, max_len):
        unk_id = word_to_id[self.UNK_TOKEN]
        pad_id = word_to_id[self.PAD_TOKEN]
        ids = [word_to_id.get(word, unk_id) for word in tokens[:max_len]]
        length = len(ids)
        if length < max_len:
            ids.extend([pad_id] * (max_len - length))
        return ids, max(length, 1)

    def _load_reviews(self, root, split, label_name, label_id, limit):
        folder = os.path.join(root, split, label_name)
        paths = [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if name.endswith('.txt')
        ]
        if limit is not None:
            paths = paths[:limit]

        texts = []
        labels = []
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(self._clean_text(f.read()))
            labels.append(label_id)
        return texts, labels

    def _load_classification(self):
        print('loading stage 4 text classification data...')
        root = self._resolve_path()

        per_class_train_limit = None
        per_class_test_limit = None
        if self.max_train_samples is not None:
            per_class_train_limit = max(1, self.max_train_samples // 2)
        if self.max_test_samples is not None:
            per_class_test_limit = max(1, self.max_test_samples // 2)

        train_pos, train_pos_y = self._load_reviews(root, 'train', 'pos', 1, per_class_train_limit)
        train_neg, train_neg_y = self._load_reviews(root, 'train', 'neg', 0, per_class_train_limit)
        test_pos, test_pos_y = self._load_reviews(root, 'test', 'pos', 1, per_class_test_limit)
        test_neg, test_neg_y = self._load_reviews(root, 'test', 'neg', 0, per_class_test_limit)

        train_tokens = train_pos + train_neg
        train_y = train_pos_y + train_neg_y
        test_tokens = test_pos + test_neg
        test_y = test_pos_y + test_neg_y

        word_to_id, id_to_word, vocab_counter = self._build_vocab(
            train_tokens,
            self.max_vocab_size,
            self.min_freq,
            [self.PAD_TOKEN, self.UNK_TOKEN],
        )

        train_X, train_lengths = zip(*[
            self._encode_and_pad(tokens, word_to_id, self.max_len)
            for tokens in train_tokens
        ])
        test_X, test_lengths = zip(*[
            self._encode_and_pad(tokens, word_to_id, self.max_len)
            for tokens in test_tokens
        ])

        return {
            'task': 'classification',
            'train_X': np.asarray(train_X, dtype=np.int64),
            'train_y': np.asarray(train_y, dtype=np.int64),
            'train_lengths': np.asarray(train_lengths, dtype=np.int64),
            'test_X': np.asarray(test_X, dtype=np.int64),
            'test_y': np.asarray(test_y, dtype=np.int64),
            'test_lengths': np.asarray(test_lengths, dtype=np.int64),
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'vocab_size': len(word_to_id),
            'vocab_counter': vocab_counter,
            'max_len': self.max_len,
        }

    def _read_jokes(self, path):
        jokes = []
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('Joke') or row.get('joke') or ''
                tokens = self._clean_text(text)
                if len(tokens) >= self.context_len:
                    jokes.append(tokens + [self.EOS_TOKEN])
        return jokes

    def _load_generation(self):
        print('loading stage 4 text generation data...')
        path = self._resolve_path()
        jokes = self._read_jokes(path)

        word_to_id, id_to_word, vocab_counter = self._build_vocab(
            jokes,
            self.max_vocab_size,
            1,
            [self.PAD_TOKEN, self.UNK_TOKEN, self.EOS_TOKEN],
        )

        unk_id = word_to_id[self.UNK_TOKEN]
        examples_X = []
        examples_y = []
        for tokens in jokes:
            ids = [word_to_id.get(word, unk_id) for word in tokens]
            for i in range(0, len(ids) - self.context_len):
                examples_X.append(ids[i:i + self.context_len])
                examples_y.append(ids[i + self.context_len])

        pairs = list(zip(examples_X, examples_y))
        random.Random(self.random_seed).shuffle(pairs)
        if self.max_train_samples is not None:
            pairs = pairs[:self.max_train_samples]

        split_at = int(len(pairs) * (1.0 - self.validation_ratio))
        split_at = max(1, min(split_at, len(pairs) - 1))
        train_pairs = pairs[:split_at]
        test_pairs = pairs[split_at:]

        train_X, train_y = zip(*train_pairs)
        test_X, test_y = zip(*test_pairs)

        seed_prompts = self._common_seed_prompts(jokes)

        return {
            'task': 'generation',
            'train_X': np.asarray(train_X, dtype=np.int64),
            'train_y': np.asarray(train_y, dtype=np.int64),
            'test_X': np.asarray(test_X, dtype=np.int64),
            'test_y': np.asarray(test_y, dtype=np.int64),
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'vocab_size': len(word_to_id),
            'vocab_counter': vocab_counter,
            'context_len': self.context_len,
            'seed_prompts': seed_prompts,
            'jokes': jokes,
        }

    def _common_seed_prompts(self, jokes):
        starts = Counter()
        for tokens in jokes:
            if len(tokens) >= self.context_len:
                starts[tuple(tokens[:self.context_len])] += 1
        prompts = [' '.join(words) for words, _ in starts.most_common(3)]
        for prompt in ['what did the', 'why did the', 'what do you']:
            if prompt not in prompts:
                prompts.append(prompt)
        return prompts[:4]
