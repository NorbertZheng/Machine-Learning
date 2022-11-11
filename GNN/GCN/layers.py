#!/usr/bin/env python3
"""
Created on 12:49, Oct. 31st, 2022

@author: Norbert Zheng
"""
import tensorflow as tf
# local dep
import inits

__all__ = [
    "get_layer_uid",
    "sparse_dropout",
    "dot",
    "Dense",
    "GraphConvolution",
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
    Dropout for sparse tensors.
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / (1 - rate))

# def dot func
def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs. dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

# def Dense class
class Dense(tf.keras.layers.Layer):
    """
    Dense layer.
    """

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
        act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        """
        Initialize `Dense` class.
        """
        # Initialize super to inherit Layer-style class.
        super(Dense, self).__init__(**kwargs)

        ## Initialize parameters.
        # Use dropout to specify whether to use placeholders.dropout.
        if dropout:
            self.dropout = placeholders["dropout"]
        else:
            self.dropout = 0.
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        # Helper variable for sparse dropout.
        self.num_features_nonzero = placeholder["num_features_nonzero"]

        # Initialize variables.
        with tf.variable_scope(self.name + "_vars"):
            self.vars["weights"] = inits.glorot([input_dim, output_dim], name="weights")
            if self.bias: self.vars["bias"] = inits.zeros([output_dim,], name="bias")

        # Check whether log current layer.
        if self.logging: self._log_vars()

    # def _call func
    def _call(self, inputs):
        """
        Forward `Dense` layer.
        """
        # Copy inputs to x.
        x = inputs
        # Use dropout to update x.
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)
        # Transform x to get the output.
        output = dot(x, self.vars["weights"], sparse=self.sparse_inputs)
        if self.bias: output += self.vars["bias"]
        # Return the final output.
        return self.act(output)

# def GraphConvolution class
class GraphConvolution(tf.keras.layers.Layer):
    """
    Graph convolution layer.
    """

    def __init__(self, input_dim, output_dim, num_features_nonzero, dropout=0., is_sparse_inputs=False,
        activation=tf.nn.relu, bias=False, featureless=False, **kwargs):
        """
        Initialize `GraphConvolution` class.
        """
        # Initialize super to inherit Layer-style class.
        super(GraphConvolution, self).__init__(**kwargs)

        # Initialize parameters.
        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = num_features_nonzero

        # Initialize variables.
        self.weights_ = []
        for i in range(1):
            w = self.add_weight("weight" + str(i), [input_dim, output_dim])
            self.weights_.append(w)
        if self.bias: self.bias = self.add_weight("bias", [output_dim,])

    # def call func
    def call(self, inputs, training=None):
        """
        Forward `GraphConvolution` layer.
        """
        # Copy inputs to x.
        x, support_ = inputs
        # Use dropout to update x. Only dropout when not training.
        if training is not False and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif training is not False:
            x = tf.nn.dropout(x, self.dropout)
        # Excute convolution.
        supports = list()
        for i in range(len(support_)):
            # If it has features x.
            if not self.featureless:
                pre_sup = dot(x, self.weights_[i], sparse=self.is_sparse_inputs)
            else:
                pre_sup = self.weights_[i]
            support = dot(support_[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        if self.bias: output += self.bias
        # Return the final output.
        return self.activation(output)

