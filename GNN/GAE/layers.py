#!/usr/bin/env python3
"""
Created on 21:36, Nov. 9th, 2022

@author: Norbert Zheng
"""
import tensorflow as tf
# local dep
import inits

__all__ = [
    "get_layer_uid",
    "sparse_dropout",
    "GraphConvolution",
    "GraphConvolutionSparse",
    "InnerProductDecoder",
]

# Global unique layer ID dictionary for layer name assignment.
_LAYER_UIDS = {}

# def get_layer_uid func
def get_layer_uid(layer_name=""):
    """
    Helper function, assigns unique layer IDs.
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

# def sparse_dropout func
def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements).
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / (1 - rate))

"""
Common classes.
"""
# def GraphConvolution class
class GraphConvolution(tf.keras.layers.Layer):
    """
    Basic graph convolution layer for undirected graph without edge labels.
    """

    def __init__(self, input_dim, output_dim, adjacency, n_x_sparse=None, activation=tf.nn.relu):
        """
        Initialize `GraphConvolution` object.
        :param input_dim: int - The dimension of input data, not considering `batch_size` (e.g. n_nodes).
        :param output_dim: int - The dimension of output data, not considering `batch_size` (e.g. n_nodes).
        :param adjacency: (3[tuple],) - The tuple representation of the adjacency matrix of the input graph.
            We should note that adjacency is SparseTensor, we should use `tf.sparse` to operate it.
        :param n_x_sparse: int - The total number of non-zero features in sparse matrix.
        :param activation: func - The type of activation function.
        """
        # Initialize super to inherit `tf.keras.layers.Layer`-style class.
        super(GraphConvolution, self).__init__()
        # Initialize parameters.
        self.adjacency = adjacency
        self.n_x_sparse = n_x_sparse
        self.activation = activation
        # Initialize variables.
        self.weights_ = inits.glorot((input_dim, output_dim), name="weights")

    # def call func
    def call(self, inputs, dropout=0.):
        """
        Forward GraphConvolution Layer.
        :param inputs: (n_nodes, input_dim) - The input features (or embeddings).
            If we enables `featureless` in `Model`, inputs should be identity matrix.
        :param dropout: int - The rate of dropout.
        :return outputs: (n_nodes, output_dim) - The output embeddings.
        """
        return self._call_basic(inputs, dropout) if self.n_x_sparse is None else self._call_sparse(inputs, dropout)

    # def _call_basic func
    def _call_basic(self, inputs, dropout=0.):
        """
        Forward GraphConvolution Layer, and inputs is not SparseTensor.
        :param inputs: (n_nodes, input_dim) - The input features (or embeddings).
            If we enables `featureless` in `Model`, inputs should be identity matrix.
        :param dropout: int - The rate of dropout.
        :return outputs: (n_nodes, output_dim) - The output embeddings.
        """
        # Copy feature inputs to x, and then dropout.
        # x - (n_nodes, input_dim)
        x = tf.nn.dropout(inputs, dropout)
        # Multiply weights to transform x, and then use adjacency matrix to spread.
        # x - (n_nodes, output_dim)
        x = tf.sparse.sparse_dense_matmul(self.adjacency, tf.matmul(x, self.weights_))
        # Return the final outputs.
        return self.activation(x)

    # def _call_sparse func
    def _call_sparse(self, inputs, dropout=0.):
        """
        Forward GraphConvolution Layer, and inputs is SparseTensor.
        :param inputs: (n_nodes, input_dim) - The input features (or embeddings).
            If we enables `featureless` in `Model`, inputs should be identity matrix.
        :param dropout: int - The rate of dropout.
        :return outputs: (n_nodes, output_dim) - The output embeddings.
        """
        # Copy inputs to x, and then dropout.
        # x - (n_nodes, input_dim)
        x = sparse_dropout(inputs, dropout, (self.n_x_sparse,))
        # Multiply weights to transform x, and then use adjacency matrix to spread.
        # x - (n_nodes, output_dim)
        x = tf.sparse.sparse_dense_matmul(self.adjacency, tf.sparse.sparse_dense_matmul(x, self.weights_))
        # Return the final outputs.
        return self.activation(x)

"""
GAE related classes.
"""
# def InnerProductDecoder class.
class InnerProductDecoder(tf.keras.layers.Layer):
    """
    Decoder model layer for link prediction, e.g. adjacency matrix.
    """

    def __init__(self, input_dim, dropout=0., activation=tf.nn.sigmoid):
        """
        Initialize `InnerProductDecoder` object.
        :param input_dim: int - The dimension of input data, not considering `batch_size` (e.g. n_nodes).
        :param dropout: int - The rate of dropout.
        :param activation: func - The type of activation function.
        """
        # Initialize super to inherit `Layer`-style class.
        super(InnerProductDecoder, self).__init__()
        # Initialize parameters.
        self.dropout = dropout
        self.activation = activation

    # def call func
    def call(self, inputs):
        """
        Forward InnerProductDecoder Layer.
        :param inputs: (n_nodes, input_dim) - The input data.
        :return outputs: (n_nodes*n_nodes,) - The predicted adjacency matrix.
        """
        # Copy inputs to z, and then dropout.
        # z - (n_nodes, input_dim)
        z = tf.nn.dropout(inputs, self.dropout)
        # Calculate the predicted adjacency matrix using inner product of z.
        # a_hat - (n_nodes*n_nodes,)
        a_hat = tf.reshape(tf.matmul(z, tf.transpose(z)), (-1,))
        # Return the final outputs.
        return self.activation(a_hat)

if __name__ == "__main__":
    import numpy as np
    import scipy.sparse as sp
    # local dep
    import configs, utils

    # Load data. We should note that the diagonal of adj is 0s.
    # adj - (n_nodes, n_nodes)
    # features - (n_nodes, n_x)
    adj, features = utils.load_data(configs.args.dataset)
    adj, features = adj.astype(np.float32), features.astype(np.float32)
    assert np.diag(adj.toarray()).sum() == 0
    # Normalize adjacency matrix.
    # adj_norm - (n_nodes, n_nodes)
    adj_norm = tf.SparseTensor(*utils.preprocess_adj(adj.tocoo()))
    # Get the sparse tensor representation of features.
    # features - (n_nodes, n_x)
    features = utils.sparse2tuple(features.tocoo())
    n_x = features[2][1]; n_x_sparse = features[1].shape[0]
    features = tf.SparseTensor(*features)

    # Instantiate GraphConvolution (sparse) layer.
    # h1_dense - (n_nodes, 128)
    gc_sparse_inst = GraphConvolution(
        input_dim=n_x,
        output_dim=128,
        adjacency=adj_norm,
        n_x_sparse=n_x_sparse
    )
    h1_dense = gc_sparse_inst(features)
    # Instantiate GraphConvolution (dense) layer.
    # z_dense - (n_nodes, 2)
    gc_dense_inst = GraphConvolution(
        input_dim=h1_dense.shape[1],
        output_dim=2,
        adjacency=adj_norm
    )
    z_dense = gc_dense_inst(h1_dense)
    # Instantiate InnerProductDecoder layer.
    # a_dense - (n_nodes, n_nodes)
    ipd_inst = InnerProductDecoder(input_dim=z_dense.shape[1])
    a_dense = ipd_inst(z_dense)

