#!/usr/bin/env python3
"""
Created on 16:06, Nov. 10th, 2022

@author: Norbert Zheng
"""

__all__ = [
    "gae_params",
    "gvae_params",
]

# def gae_params dict
gae_params = {
    ## -- Model parameters
    # The number of features corresponding to one node.
    # Use the true dataset to initialize `n_x`.
    # The total number of features that are non-zero.
    # Use the true dataset to initialize `n_x_sparse`.
    # The number of neurons in encoder hidden layers.
    "n_hidden_encoder": [32,],
    # The number of neurons in latent node representation.
    "n_z": 16,
    # The rate of dropout (1 - keep probability).
    "dropout": 0.,
    ## -- Train parameters
    # The learning rate.
    "learning_rate": 0.01,
    # The number of training epochs.
    "n_epochs": 200,
    # Whether to use features (e.g. (n_nodes, n_x) feature matrix)
    # or not (e.g. (n_nodes, n_nodes) identity matrix).
    "use_feature": False,
}

# def gvae_params dict
gvae_params = {
    ## -- Model parameters
    # The number of features corresponding to one node.
    # Use the true dataset to initialize `n_x`.
    # The total number of features that are non-zero.
    # Use the true dataset to initialize `n_x_sparse`.
    # The number of neurons in encoder hidden layers.
    "n_hidden_encoder": [32,],
    # The number of neurons in latent node representation.
    "n_z": 16,
    # The rate of dropout (1 - keep probability).
    "dropout": 0.,
    ## -- Train parameters
    # The learning rate.
    "learning_rate": 0.01,
    # The number of training epochs.
    "n_epochs": 200,
    # Whether to use features (e.g. (n_nodes, n_x) feature matrix)
    # or not (e.g. (n_nodes, n_nodes) identity matrix).
    "use_feature": False,
}

