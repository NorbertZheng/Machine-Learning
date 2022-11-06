#!/usr/bin/env python3
"""
Created on 22:04, Nov. 6th, 2022

@author: Norbert Zheng
"""
import tensorflow as tf

__all__ = [
    "default_params",
]

# def default_params dict
default_params = {
    ## -- Model parameters
    # The activation type of Dense.
    "activation": tf.nn.relu,
    # The dimension of input x, e.g. MNIST data input (img shape: 28*28).
    "n_x": 784,
    # The dimension of latent state z.
    "n_z": 2,
    ## -- Train parameters
    # The learning rate.
    "learning_rate": 0.001,
    # The size of batch.
    "batch_size": 100,
    # The number of training epochs.
    "n_epochs": 75,
    # The period of log.
    "n_log": 5,
    # The number of samples used in reconstruction process.
    "n_samples_reconstr": 5,
    # The number of samples used in recognition process.
    "n_samples_recog": 5000,
}

