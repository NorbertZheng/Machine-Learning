#!/usr/bin/env python3
"""
Created on 15:17, Oct. 31st, 2022

@author: Norbert Zheng
"""
import tensorflow as tf
# local dep
import configs, layers, metrics

__all__ = [
    "MLP",
    "GCN",
]

# def MLP class
class MLP(tf.keras.Model):
    """
    Multi-layer Perceptron.
    """

    def __init__(self, placeholders, input_dim, **kwargs):
        """
        Initialize `MLP` class.
        """
        # Initialize super to inherit Model-style class.
        super(MLP, self).__init__(**kwargs)

        # Initialize parameters.
        self.inputs = placeholders["features"]
        self.input_dim = input_dim
        self.output_dim = placeholder["label"].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=configs.args.learning_rate)

        # Initialize variables.
        self.build()

    # def _loss func
    def _loss(self):
        """
        Calculate loss, including prediction loss and weight decay loss.
        """
        # Calculate weight decay loss.
        for var in self.layers[0].vars.values():
            self.loss += configs.args.weight_decay * tf.nn.l2_loss(var)
        # Calculate entropy error.
        self.loss += metrics.masked_softmax_cross_entropy(self.outputs,
            self.placeholders["labels"], self.placeholders["label_mask"])

    # def _accuracy func
    def _accuracy(self):
        """
        Calculate accuracy.
        """
        self.accuracy = metrics.masked_accuracy(self.outputs,
            self.placeholders["labels"], self.placeholders["labels_mask"])

    # def _build func
    def _build(self):
        """
        Build the variables of network.
        """
        self.layers.append(layers.Dense(input_dim=input_dim, output_dim=configs.args.hidden1,
            placeholders=self.placeholders, act=tf.nn.relu, dropout=True, sparse_inputs=True, logging=self.logging))
        self.layers.append(layers.Dense(input_dim=configs.args.hidden1, output_dim=self.output_dim,
            placeholders=self.placeholders, act=lambda x: x, dropout=True, logging=self.logging))

    # def predict func
    def predict(self):
        """
        Use softmax to activate the output, which is already calculated.
        """
        return tf.nn.softmax(self.outputs)

# def GCN class
class GCN(tf.keras.Model):
    """
    Graph Convolution Network.
    """

    def __init__(self, input_dim, output_dim, num_features_nonzero, **kwargs):
        """
        Initialize `GCN` class.
        """
        # Initialize super to inherit Model-style class.
        super(GCN, self).__init__(**kwargs)

        # Initialize parameters.
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize variables.
        self.layers_ = []
        self.layers_.append(layers.GraphConvolution(input_dim=self.input_dim, output_dim=configs.args.hidden1,
            num_features_nonzero=num_features_nonzero, activation=tf.nn.relu,
            dropout=configs.args.dropout, is_sparse_inputs=True))
        self.layers_.append(layers.GraphConvolution(input_dim=configs.args.hidden1, output_dim=self.output_dim,
            num_features_nonzero=num_features_nonzero, activation=lambda x: x, dropout=configs.args.dropout))

    # def call func
    def call(self, inputs, training=None):
        """
        Forward `GCN` network.
        """
        # Copy inputs to x...
        x, label, mask, support = inputs
        outputs = [x,]
        # Use layers to calculate outputs.
        for layer in self.layers:
            hidden = layer((outputs[-1], support), training=training)
            outputs.append(hidden)
        output = outputs[-1]
        # Calculate loss.
        loss = tf.zeros([])
        for var in self.layers_[0].trainable_variables:
            loss += configs.args.weight_decay * tf.nn.l2_loss(var)
        loss += metrics.masked_softmax_cross_entropy(output, label, mask)
        acc = metrics.masked_accuracy(output, label, mask)
        # Return the final loss & acc.
        return loss, acc

    # def predict func
    def predict(self):
        """
        Use softmax to activate the output, which is already calculated.
        """
        return tf.nn.softmax(self.outputs)

