#!/usr/bin/env python3
"""
Created on 16:20, Nov. 10th, 2022

@author: Norbert Zheng
"""
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

__all__ = [
    "parse_index_file",
    "load_data",
    "sparse2tuple",
    "normalize_adj",
    "preprocess_adj",
    "prepare_train",
]

"""
data-load related functions.
"""
# def parse_index_file func
def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# def load_data func
def load_data(dataset):
    """
    Load input data from gcn/data directory.
    ind.dataset.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object.
    All objects above must be saved using python pickle module.
    """
    names = ["x", "tx", "allx", "graph"]
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position.
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

"""
data-preprocess related functions.
"""
# def sparse2tuple func
def sparse2tuple(sparse_mat):
    """
    Convert sparse matrix to tuple representation.
    :param sparse_mat: sp.coo_matrix - The sparse matrix (or sparse matrix list).
    :return sparse_tuple: (3[tuple],) - The tuple representation of sparse matrix (or sparse matrix list).
    """
    def to_tuple(matrix):
        if not sp.isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        coords = np.vstack((matrix.row, matrix.col)).transpose()
        values = matrix.data
        shape = matrix.shape
        return coords, values, shape

    if isinstance(sparse_mat, list):
        sparse_tuple = []
        for i in range(len(sparse_mat)):
            sparse_tuple.append(to_tuple(sparse_mat[i]))
    else:
        sparse_tuple = to_tuple(sparse_mat)

    return sparse_tuple

# def tuple2sparse func
def tuple2sparse(sparse_tuple):
    """
    Convert tuple representation to sparse matrix.
    :param sparse_tuple: (3[tuple],) - The tuple representation of sparse matrix (or sparse matrix list).
    :return sparse_mat: sp.coo_matrix - The sparse matrix (or sparse matrix list) of tuple representation.
    """
    if isinstance(sparse_tuple, list):
        sparse_mat = []
        for i in range(len(sparse_tuple)):
            coords_i, values_i, shape_i = sparse_tuple[i]
            sparse_mat.append(sp.coo_matrix((values_i, coords_i.T), shape=shape_i))
    else:
        coords, values, shape = sparse_tuple
        sparse_mat = sp.coo_matrix((values, coords.T), shape=shape)
    return sparse_mat

# def normalize_adj func
def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    :param adj: (n_nodes, n_nodes) - The adjacency matrix with diagonal as 0s.
    :return adj_norm: (n_nodes, n_nodes) - The normalized adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# def preprocess_adj func
def preprocess_adj(adj):
    """
    Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    :param adj: (n_nodes, n_nodes) - The adjacency matrix with diagonal as 0s.
    :param adj_norm: (n_nodes, n_nodes) - The normalized adjacency matrix.
    """
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0], dtype=np.float32))
    return adj_norm

# def prepare_train func
def prepare_train(adj):
    """
    Build [train,val,test] set, where test set has 10% positive links.
    Note: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    """
    # Remove disgonal elements.
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis,:], [0]), shape=adj.shape); adj.eliminate_zeros()
    # Check that diagonal is zero.
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse2tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse2tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    test_edges_false = np.array(test_edges_false)

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    val_edges_false = np.array(val_edges_false)

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


