'''
Concrete IO class for the Stage 5 citation-network datasets (Cora / Citeseer / Pubmed).

This loader reads the raw <node> and <link> files, builds the symmetric
normalized adjacency matrix used by a GCN (D^-1/2 (A + I) D^-1/2), row-normalizes
the node features, and draws a class-balanced train / validation / test split
following the partition sizes specified in the Stage 5 ReadMe.
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD
# Extended for ECS 170 Stage 5 (GCN node classification).

from local_code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp


class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    # Per-dataset class-balanced split sizes (number of nodes PER CLASS),
    # taken directly from the Stage 5 ReadMe instructions.
    #   cora:     7 classes -> 20 train / 150 test per class (140 / 1050 total)
    #   citeseer: 6 classes -> 20 train / 200 test per class (120 / 1200 total)
    #   pubmed:   3 classes -> 20 train / 200 test per class (60  / 600  total)
    sampling_config = {
        'cora':     {'train_per_class': 20, 'test_per_class': 150},
        'citeseer': {'train_per_class': 20, 'test_per_class': 200},
        'pubmed':   {'train_per_class': 20, 'test_per_class': 200},
        # cora-small is a tiny hand-crafted toy graph kept for debugging.
        'cora-small': {'train_per_class': 1, 'test_per_class': 1},
    }
    # Held-out validation nodes per class (used only for early stopping /
    # model selection).  A large class-balanced validation set mirrors the
    # balanced test set, so validation metrics are a faithful and low-variance
    # proxy for the test metrics during model / seed selection.  Classes with
    # fewer leftover nodes simply contribute as many as are available.
    val_per_class = 200

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)
        # seed controls the random train/val/test sampling for reproducibility
        self.seed = seed

    def adj_normalize(self, mx):
        """Symmetrically normalize a sparse adjacency matrix: D^-1/2 * A * D^-1/2."""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def normalize_features(self, mx):
        """Row-normalize a sparse feature matrix so each node's features sum to 1."""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse COO tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def encode_onehot(self, labels):
        """Deterministically one-hot encode string labels (classes sorted for reproducibility)."""
        classes = sorted(set(labels))
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels, classes

    def _balanced_split(self, labels):
        """Draw class-balanced train / validation / test splits per the ReadMe.

        For every class we sample (in this order, disjointly) the required number
        of train nodes, then test nodes, then validation nodes.  Train and test
        sizes follow the ReadMe; the validation set is class-balanced too so that
        validation metrics track the balanced test metrics during model selection.

        Returns numpy arrays of node indices for (train, val, test).
        """
        rng = np.random.default_rng(self.seed)
        config = self.sampling_config.get(self.dataset_name,
                                          {'train_per_class': 20, 'test_per_class': 200})
        n_train = config['train_per_class']
        n_test = config['test_per_class']

        num_classes = int(labels.max()) + 1
        train_idx, val_idx, test_idx = [], [], []
        for c in range(num_classes):
            class_members = np.where(labels == c)[0]
            rng.shuffle(class_members)
            available = len(class_members)
            take_train = min(n_train, available)
            take_test = min(n_test, max(0, available - take_train))
            start_val = take_train + take_test
            take_val = min(self.val_per_class, max(0, available - start_val))

            train_idx.extend(class_members[:take_train].tolist())
            test_idx.extend(class_members[take_train:start_val].tolist())
            val_idx.extend(class_members[start_val:start_val + take_val].tolist())

        train_idx = np.array(sorted(train_idx), dtype=np.int64)
        val_idx = np.array(sorted(val_idx), dtype=np.int64)
        test_idx = np.array(sorted(test_idx), dtype=np.int64)

        return train_idx, val_idx, test_idx

    def load(self):
        """Load a citation-network dataset and return the graph plus the index split."""
        print('Loading {} dataset...'.format(self.dataset_name))

        # ---- load node data: <node_id> <features...> <label> ----
        idx_features_labels = np.genfromtxt(
            "{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        features = self.normalize_features(features)
        onehot_labels, class_names = self.encode_onehot(idx_features_labels[:, -1])

        # ---- load link data and build the (undirected, self-looped) graph ----
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(
            "{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(onehot_labels.shape[0], onehot_labels.shape[0]),
                            dtype=np.float32)
        # symmetrize the directed citation links so information flows both ways
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # ---- convert everything to torch tensors ----
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # ---- class-balanced train / val / test split (ReadMe partition sizes) ----
        labels_np = labels.numpy()
        idx_train, idx_val, idx_test = self._balanced_split(labels_np)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        print('  nodes: {}, features: {}, classes: {}'.format(
            features.shape[0], features.shape[1], len(class_names)))
        print('  split -> train: {}, val: {}, test: {}'.format(
            len(idx_train), len(idx_val), len(idx_test)))

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {
            'node': idx_map,
            'edge': edges,
            'X': features,
            'y': labels,
            'utility': {'A': adj, 'reverse_idx': reverse_idx_map},
            'class_names': class_names,
        }
        return {'graph': graph, 'train_test_val': train_test_val}
