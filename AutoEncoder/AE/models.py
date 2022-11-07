#!/usr/bin/env python3
"""
Created on 22:10, Nov. 6th, 2022

@author: Norbert Zheng
"""
import copy as cp
import numpy as np
import tensorflow as tf
# local dep
import inits

__all__ = [
    "AE",
]

# def AE class
class AE(tf.keras.Model):
    """
    Auto-Encoder (AE) implemented using TensorFlow.
    This implementation uses deterministic encoders and decoders and realized
    by multi-layer perceptrons. The AE can be learned end-to-end.
    """

    def __init__(self, params):
        """
        Initialize `AE` object.
        :param params: The parameters used to initialize AE.
        """
        # Initialize super to inherit `tf.keras.Model`-style class.
        super(AE, self).__init__()
        # Initialize parameters.
        self.params = cp.deepcopy(params)
        # Initialize variables.
        self._init_trainable()

    # def _init_trainable func
    def _init_trainable(self):
        """
        Initialize trainable variables.
        """
        ## Initialize weights_.
        self.weights_ = dict()
        ## -- Recognition network.
        self.weights_["recognition"] = dict()
        # Initialize weights of recognition network.
        self.weights_["recognition"]["weights"] = {
            "h1": tf.Variable(inits.xavier((self.params["n_x"], self.params["n_hidden_recog_1"]))),
            "h2": tf.Variable(inits.xavier((self.params["n_hidden_recog_1"], self.params["n_hidden_recog_2"]))),
            "out": tf.Variable(inits.xavier((self.params["n_hidden_recog_2"], self.params["n_z"]))),
        }
        # Initialize biases of recognition network.
        self.weights_["recognition"]["biases"] = {
            "h1": tf.Variable(tf.zeros((self.params["n_hidden_recog_1"],), dtype=tf.float32)),
            "h2": tf.Variable(tf.zeros((self.params["n_hidden_recog_2"],), dtype=tf.float32)),
            "out": tf.Variable(tf.zeros((self.params["n_z"],), dtype=tf.float32)),
        }
        ## -- Generation network.
        self.weights_["generation"] = dict()
        # Initialize weights of generation network.
        self.weights_["generation"]["weights"] = {
            "h1": tf.Variable(inits.xavier((self.params["n_z"], self.params["n_hidden_gener_1"]))),
            "h2": tf.Variable(inits.xavier((self.params["n_hidden_gener_1"], self.params["n_hidden_gener_2"]))),
            "out": tf.Variable(inits.xavier((self.params["n_hidden_gener_2"], self.params["n_x"]))),
        }
        # Initialize biases of generation network.
        self.weights_["generation"]["biases"] = {
            "h1": tf.Variable(tf.zeros((self.params["n_hidden_gener_1"],), dtype=tf.float32)),
            "h2": tf.Variable(tf.zeros((self.params["n_hidden_gener_2"],), dtype=tf.float32)),
            "out": tf.Variable(tf.zeros((self.params["n_x"],), dtype=tf.float32)),
        }

    # def call func
    def call(self, X, training=None, mask=None):
        """
        Forward layers in `AE` to get the final result.
        :param X: (batch_size, n_x) - The original x.
        :param training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
        :param mask: A mask or list of masks. A mask can be either a tensor or None (no mask).
        :return X_reconstr: (batch_size, n_x) - The reconstructed x.
        :return loss: int - The whole loss, including reconstructed loss.
        """
        # Use recognition network to determine the latent state.
        # z - (batch_size, n_z)
        z = self._recognition(X)
        # Use generation network to determine mean of Bernoulli distribution of reconstructed x.
        # x_reconstr - (batch_size, n_x)
        x_reconstr = self._generation(z)
        # Calculate the whole loss, including reconstructed loss.
        # loss - int
        loss = self._loss(X, x_reconstr)
        # Return the final X_reconstr & loss.
        return x_reconstr, loss

    # def recognite func
    def recognite(self, X):
        """
        Recognite data by mapping it into the latent space.
        :param X: (batch_size, n_x) - The original x.
        :return Z: (batch_size, n_z) - The corresponding latent state z.
        """
        # z - (batch_size, n_z)
        z = self._recognition(X)
        # Return the final Z.
        return z

    # def generate func
    def generate(self, Z=None):
        """
        Generate data by sampling from latent space. If Z is not None, data for this point in latent space
        is generated. Otherwise, Z is drawn from prior in latent space.
        :param Z: (batch_size, n_z) - The given latent state z.
        :return X: (batch_size, n_x) - The generated x.
        """
        # If Z is None, draw Z from prior in latent space.
        Z = np.random.normal((self.params["n_z"],)) if Z is None else Z
        # x_reconstr - (batch_size, n_x)
        x_reconstr = self._generation(Z)
        # Return the final X.
        return x_reconstr

    # def _recognition func
    def _recognition(self, x):
        """
        Generate deterministic encoder (recognition network), which maps inputs onto a latent
        state in latent space. The transformation is parameterized and can be learned.
        :param x: (batch_size, n_x) - The original x.
        :return z: (batch_size, n_z) - The corresponding latent state z.
        """
        # Forward recognition network.
        # layer1 - (batch_size, n_hidden_recog_1)
        layer1 = self.params["activation"](tf.add(
            tf.matmul(x, self.weights_["recognition"]["weights"]["h1"]),
            self.weights_["recognition"]["biases"]["h1"]
        ))
        # layer2 - (batch_size, n_hidden_recog_2)
        layer2 = self.params["activation"](tf.add(
            tf.matmul(layer1, self.weights_["recognition"]["weights"]["h2"]),
            self.weights_["recognition"]["biases"]["h2"]
        ))
        # z - (batch_size, n_z)
        z = self.params["activation"](tf.add(
            tf.matmul(layer2, self.weights_["recognition"]["weights"]["out"]),
            self.weights_["recognition"]["biases"]["out"]
        ))
        # Return the final z.
        return z

    # def _generation func
    def _generation(self, z):
        """
        Generate deterministic decoder (decoder network), which maps points in latent space onto
        data points in data space. The transformation is parameterized and can be learned.
        :param z: (batch_size, n_z) - The given latent state z.
        :return x_reconstr: (batch_size, n_x) - The reconstructed x.
        """
        # Forward generation network.
        # layer1 - (batch_size, n_hidden_gener_1)
        layer1 = self.params["activation"](tf.add(
            tf.matmul(z, self.weights_["generation"]["weights"]["h1"]),
            self.weights_["generation"]["biases"]["h1"]
        ))
        # layer2 - (batch_size, n_hidden_gener_2)
        layer2 = self.params["activation"](tf.add(
            tf.matmul(layer1, self.weights_["generation"]["weights"]["h2"]),
            self.weights_["generation"]["biases"]["h2"]
        ))
        # x_reconstr - (batch_size, n_x)
        x_reconstr = self.params["activation"](tf.add(
            tf.matmul(layer2, self.weights_["generation"]["weights"]["out"]),
            self.weights_["generation"]["biases"]["out"]
        ))
        # Return the final x_reconstr.
        return x_reconstr

    # def _loss func
    def _loss(self, x, x_reconstr):
        """
        Calculate the whole loss, including reconstruction loss and latent loss.
        :param x: (batch_size, n_x) - The original x.
        :param x_reconstr: (batch_size, n_x) - The reconstructed x.
        :return loss: int - The whole loss, including reconstructed loss and latent loss.
        """
        # The loss is only composed of one term:
        # 1.) The reconstruction loss (the mean square error between x and x_reconstr).
        # reconstr_loss - (batch_size,)
        reconstr_loss = 0.5 * tf.reduce_sum(tf.square(x-x_reconstr), axis=-1)
        # Calculate the whole loss, averaging over batch.
        # loss - int
        loss = tf.reduce_mean(reconstr_loss)
        # Return the final loss.
        return loss

