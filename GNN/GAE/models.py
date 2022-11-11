#!/usr/bin/env python3
"""
Created on 12:06, Nov. 10th, 2022

@author: Norbert Zheng
"""
import math
import copy as cp
import tensorflow as tf
import scipy.sparse as sp
# local dep
import params, layers, utils

__all__ = [
    "GAE",
    "GVAE",
]

# def GAE class.
class GAE(tf.keras.Model):
    """
    Graph Auto-Encoder (GAE).
    """

    def __init__(self, params, adjacency):
        """
        Initialize `GAE` object.
        :param params: The parameters used to initialize `GAE` object.
        :param adjacency: (3[list],) - The adjacency matrix of the whole graph with diagonal as 0s.
        """
        # Initialize super to inherit `tf.keras.Model`-style class.
        super(GAE, self).__init__()
        # Initialize paramters.
        # Note: adjacency includes [adj,adj_label,adj_norm].
        self.params = cp.deepcopy(params)
        self.adjacency = cp.deepcopy(adjacency)
        self.pos_weight = float(math.prod(self.adjacency[0][2]) - self.adjacency[0][0].shape[0]) / self.adjacency[0][0].shape[0]
        self.norm_loss = math.prod(self.adjacency[0][2]) /\
            float((math.prod(self.adjacency[0][2]) - self.adjacency[0][0].shape[0]) * 2)
        self.adj_label = tf.cast(utils.tuple2sparse(self.adjacency[1]).todense(), dtype=tf.float32)
        self.adj_norm = tf.cast(tf.SparseTensor(*self.adjacency[2]), dtype=tf.float32)
        # Initialize variables.
        self._init_trainable()

    # def _init_trainable func
    def _init_trainable(self):
        """
        Initialize trainable variables.
        """
        # Initialize encoder.
        # encoder - ((n_nodes, n_x) -> (n_nodes, n_z))
        self.encoder = [
            # The first layer of encoder is always sparse layer.
            layers.GraphConvolution(
                input_dim=self.params["n_x"],
                output_dim=self.params["n_hidden_encoder"][0],
                adjacency=self.adj_norm,
                n_x_sparse=self.params["n_x_sparse"],
                activation=tf.nn.relu
            ),
            # Transform the embedding of the first layer to the final node representation.
            layers.GraphConvolution(
                input_dim=self.params["n_hidden_encoder"][0],
                output_dim=self.params["n_z"],
                adjacency=self.adj_norm,
                activation=lambda x: x
            ),
        ]
        # Initialize decoder.
        # decoder - ((n_nodes, n_z) -> (n_nodes*n_nodes,))
        self.decoder = layers.InnerProductDecoder(
            input_dim=self.params["n_z"],
            activation=lambda x: x
        )

    # def call func
    def call(self, X, dropout=None, training=None, mask=None):
        """
        Forward layers in `GAE` to get the final A & loss.
        :param X: (n_nodes, n_x) - The feature matrix of the whole graph.
        :param dropout: int - The rate of dropout.
        :param training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
        :param mask: A mask or list of masks. A mask can be either a tensor or None (no mask).
        :return A: (n_nodes, n_nodes) - The reconstructed adjacency matrix.
        :return loss: int - The whole loss, including reconstructed loss.
        """
        # Use encoder to encode the original feature matrix.
        # Z - (n_nodes, n_z)
        Z = X
        for encoder_i in self.encoder:
            Z = encoder_i(Z, dropout=dropout if dropout is not None else self.params["dropout"])
        # Use decoder to decode (or reconstruct) the adjacency matrix.
        # A - (n_nodes, n_nodes)
        A = self.decoder(Z)
        # Calculate the whole loss, including reconstructed loss.
        # loss - int
        loss = self._loss(A, self.adj_label)
        # Return the final A.
        return A, loss

    # def encode func
    def encode(self, X, dropout=None):
        """
        Forward encoder layers in `GAE` to get the final Z.
        :param X: (n_nodes, n_x) - The feature matrix of the whole graph.
        :param dropout: int - The rate of dropout.
        :return Z: (n_nodes, n_z) - The latent representation of the whole graph.
        """
        # Use encoder to encode the original feature matrix.
        # Z - (n_nodes, n_z)
        Z = X
        for encoder_i in self.encoder:
            Z = encoder_i(Z, dropout=dropout if dropout is not None else self.params["dropout"])
        # Return the final Z.
        return Z

    # def _loss func
    def _loss(self, adj_pred, adj_label):
        """
        Calculate the whole loss, including reconstructed loss.
        :param adj_pred: (n_nodes, n_nodes) - The reconstructed adjacency matrix.
        :param adj_label: (n_nodes, n_nodes) - The label (or target) adjacency matrix.
        :return loss: int - The whole loss, including reconstructed loss.
        """
        # Use `weighted_cross_entropy_with_logits` to calculate loss.
        # A value pos_weight > 1 decreases the false negative count, hence
        # increasing the recall. Conversely setting pos_weight < 1 decreases
        # the false positive count and increases the precision.
        # labels * -log(sigmoid(logits)) * pos_weight +
        #     (1 - labels) * -log(1 - sigmoid(logits))
        loss = self.norm_loss * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.reshape(adj_label, (-1,)), logits=tf.reshape(adj_pred, (-1,)), pos_weight=self.pos_weight))
        # Return the final loss.
        return loss

# def GVAE class
class GVAE(GAE):
    """
    Graph Variational Auto-Encoder (GVAE).
    """

    def __init__(self, params, adjacency):
        """
        Initialize `GAE` object.
        :param params: The parameters used to initialize `GAE` object.
        :param adjacency: (3[list],) - The adjacency matrix of the whole graph with diagonal as 0s.
        """
        # Initialize super to inherit `GAE`-style class.
        super(GVAE, self).__init__(params=params, adjacency=adjacency)

    # def _init_trainable func
    def _init_trainable(self):
        """
        Initialize trainable variables.
        """
        # Initialize encoder.
        # encoder - ((n_nodes, n_x) -> (n_nodes, n_z))
        self.encoder = {
            # The first layer of encoder is always sparse layer.
            "h1": layers.GraphConvolution(
                input_dim=self.params["n_x"],
                output_dim=self.params["n_hidden_encoder"][0],
                adjacency=self.adj_norm,
                n_x_sparse=self.params["n_x_sparse"],
                activation=tf.nn.relu
            ),
            # Transform the embedding of the first layer to the mean of node representation.
            "out_mean": layers.GraphConvolution(
                input_dim=self.params["n_hidden_encoder"][0],
                output_dim=self.params["n_z"],
                adjacency=self.adj_norm,
                activation=lambda x: x
            ),
            # Transform the embedding of the first layer to the log sigma of node representation.
            "out_log_sigma": layers.GraphConvolution(
                input_dim=self.params["n_hidden_encoder"][0],
                output_dim=self.params["n_z"],
                adjacency=self.adj_norm,
                activation=lambda x: x
            ),
        }
        # Initialize decoder.
        # decoder - ((n_nodes, n_z) -> (n_nodes*n_nodes,))
        self.decoder = layers.InnerProductDecoder(
            input_dim=self.params["n_z"],
            activation=lambda x: x
        )

    # def call func
    def call(self, X, dropout=None, training=None, mask=None):
        """
        Forward layers in `GAE` to get the final A & loss.
        :param X: (n_nodes, n_x) - The feature matrix of the whole graph.
        :param dropout: int - The rate of dropout.
        :param training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
        :param mask: A mask or list of masks. A mask can be either a tensor or None (no mask).
        :return A: (n_nodes, n_nodes) - The reconstructed adjacency matrix.
        :return loss: int - The whole loss, including reconstructed loss.
        """
        # Use encoder to encode the original feature matrix.
        # Z - (n_nodes, n_z)
        h1 = self.encoder["h1"](X, dropout=dropout if dropout is not None else self.params["dropout"])
        Z_mean = self.encoder["out_mean"](h1, dropout=dropout if dropout is not None else self.params["dropout"])
        Z_log_sigma = self.encoder["out_log_sigma"](h1,
            dropout=dropout if dropout is not None else self.params["dropout"])
        Z = Z_mean + tf.random.normal(Z_log_sigma.shape) * tf.exp(Z_log_sigma)
        # Use decoder to decode (or reconstruct) the adjacency matrix.
        # A - (n_nodes, n_nodes)
        A = self.decoder(Z)
        # Calculate the whole loss, including reconstructed loss.
        # loss - int
        loss = self._loss(A, self.adj_label, Z_mean, Z_log_sigma)
        # Return the final A.
        return A, loss

    # def encode func
    def encode(self, X, dropout=None):
        """
        Forward encoder layers in `GAE` to get the final Z.
        :param X: (n_nodes, n_x) - The feature matrix of the whole graph.
        :param dropout: int - The rate of dropout.
        :return Z: (n_nodes, n_z) - The latent representation of the whole graph.
        """
        # Use encoder to encode the original feature matrix.
        # Z - (n_nodes, n_z)
        h1 = self.encoder["h1"](X, dropout=dropout if dropout is not None else self.params["dropout"])
        Z_mean = self.encoder["out_mean"](h1, dropout=dropout if dropout is not None else self.params["dropout"])
        Z_log_sigma = self.encoder["out_log_sigma"](h1,
            dropout=dropout if dropout is not None else self.params["dropout"])
        Z = Z_mean + tf.random.normal(Z_log_sigma.shape) * tf.exp(Z_log_sigma)
        # Return the final Z.
        return Z

    # def _loss func
    def _loss(self, adj_pred, adj_label, z_mean, z_log_sigma):
        """
        Calculate the whole loss, including reconstructed loss.
        :param adj_pred: (n_nodes, n_nodes) - The reconstructed adjacency matrix.
        :param adj_label: (n_nodes, n_nodes) - The label (or target) adjacency matrix.
        :param z_mean: (n_nodes, n_z) - The mean of node representation.
        :param z_log_sigma: (n_nodes, n_z) - The log sigma of node representation.
        :return loss: int - The whole loss, including reconstructed loss.
        """
        # The loss is composed of two terms:
        # 1.) The reconstruction loss. Use `weighted_cross_entropy_with_logits` to calculate loss.
        #     A value pos_weight > 1 decreases the false negative count, hence increasing the recall.
        #     Conversely setting pos_weight < 1 decreases the false positive count and increases the precision.
        #     labels * -log(sigmoid(logits)) * pos_weight + (1 - labels) * -log(1 - sigmoid(logits))
        loss_reconstr = self.norm_loss * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.reshape(adj_label, (-1,)), logits=tf.reshape(adj_pred, (-1,)), pos_weight=self.pos_weight))
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence between the distribution in latent
        #     space induced by the encoder on the data and some prior. This acts as a kind of regularizer.
        loss_latent = 1 / z_mean.shape[0] * (-0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + 2 * z_log_sigma - tf.square(z_mean) - tf.square(tf.exp(z_log_sigma)), axis=-1)))
        # Return the final loss.
        return loss_reconstr + loss_latent

if __name__ == "__main__":
    import numpy as np
    import scipy.sparse as sp
    # local dep
    import configs, params, utils

    # Load data. We should note that the diagonal of adj is 0s.
    # adj - (n_nodes, n_nodes)
    # features - (n_nodes, n_x)
    adj, features = utils.load_data(configs.args.dataset)
    adj, features = adj.astype(np.float32), features.astype(np.float32)
    assert np.diag(adj.toarray()).sum() == 0
    # Calculate the target adjacency matrix.
    # adj_label - (n_nodes, n_nodes)
    adj_label = adj + sp.eye(adj.shape[0], dtype=np.float32)
    # Normalize adjacency matrix.
    # adj_norm - (n_nodes, n_nodes)
    adj_norm = utils.preprocess_adj(adj)
    # Calculate the adjacency matrix.
    adjacency = [utils.sparse2tuple(adj), utils.sparse2tuple(adj_label), utils.sparse2tuple(adj_norm)]

    # Get the sparse tensor representation of features.
    # features - (n_nodes, n_x)
    features = utils.sparse2tuple(features)
    n_x = features[2][1]; n_x_sparse = features[1].shape[0]
    features = tf.SparseTensor(*features)

    # Instantiate GAE model.
    gae_params_inst = cp.deepcopy(params.gae_params)
    gae_params_inst["n_x"] = n_x; gae_params_inst["n_x_sparse"] = n_x_sparse
    gae_inst = GAE(params=gae_params_inst, adjacency=adjacency)
    adj_reconstr, loss = gae_inst(features)

    # Instantiate GVAE model.
    gvae_params_inst = cp.deepcopy(params.gvae_params)
    gvae_params_inst["n_x"] = n_x; gvae_params_inst["n_x_sparse"] = n_x_sparse
    gvae_inst = GVAE(params=gvae_params_inst, adjacency=adjacency)
    adj_reconstr, loss = gvae_inst(features)

