import numpy as np
import tensorflow as tf
import scipy.sparse as sp


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_one_hot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_one_hot, classes_dict


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    """
    return features(sparse), adj(sparse), labels, class_dict
    """
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    labels, class_dict = encode_onehot(idx_features_labels[:, -1])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # features.toarray()

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.
          format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.toarray(), adj, labels, class_dict


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, val_mask, test_mask


def preprocess_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
    a_hat = adj.dot(d).transpose().dot(d).toarray()
    return a_hat



