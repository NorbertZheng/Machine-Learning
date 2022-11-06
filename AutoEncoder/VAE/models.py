#!/usr/bin/env python3
"""
Created on 20:22, Nov. 5th, 2022

@author: Norbert Zheng
"""
import copy as cp
import numpy as np
import tensorflow as tf
# local dep
import inits

__all__ = [
    "VAE",
]

# def VAE class
class VAE(tf.keras.Model):
    """
    Variational Auto-Encoder (VAE) implemented using TensorFlow.
    This implementation uses probabilistic encoders and decoders using Gaussian distributions
    and realized by multi-layer perceptrons. The VAE can be learned end-to-end.
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, params):
        """
        Initialize `VAE` object.
        :param params: The parameters used to initialize VAE.
        """
        # Initialize super to inherit `tf.keras.Model`-style class.
        super(VAE, self).__init__()
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
            "out_mean": tf.Variable(inits.xavier((self.params["n_hidden_recog_2"], self.params["n_z"]))),
            "out_log_sigma": tf.Variable(inits.xavier((self.params["n_hidden_recog_2"], self.params["n_z"]))),
        }
        # Initialize biases of recognition network.
        self.weights_["recognition"]["biases"] = {
            "h1": tf.Variable(tf.zeros((self.params["n_hidden_recog_1"],), dtype=tf.float32)),
            "h2": tf.Variable(tf.zeros((self.params["n_hidden_recog_2"],), dtype=tf.float32)),
            "out_mean": tf.Variable(tf.zeros((self.params["n_z"],), dtype=tf.float32)),
            "out_log_sigma": tf.Variable(tf.zeros((self.params["n_z"],), dtype=tf.float32)),
        }
        ## -- Generation network.
        self.weights_["generation"] = dict()
        # Initialize weights of generation network.
        self.weights_["generation"]["weights"] = {
            "h1": tf.Variable(inits.xavier((self.params["n_z"], self.params["n_hidden_gener_1"]))),
            "h2": tf.Variable(inits.xavier((self.params["n_hidden_gener_1"], self.params["n_hidden_gener_2"]))),
            "out_mean": tf.Variable(inits.xavier((self.params["n_hidden_gener_2"], self.params["n_x"]))),
            "out_log_sigma": tf.Variable(inits.xavier((self.params["n_hidden_gener_2"], self.params["n_x"]))),
        }
        # Initialize biases of generation network.
        self.weights_["generation"]["biases"] = {
            "h1": tf.Variable(tf.zeros((self.params["n_hidden_gener_1"],), dtype=tf.float32)),
            "h2": tf.Variable(tf.zeros((self.params["n_hidden_gener_2"],), dtype=tf.float32)),
            "out_mean": tf.Variable(tf.zeros((self.params["n_x"],), dtype=tf.float32)),
            "out_log_sigma": tf.Variable(tf.zeros((self.params["n_x"],), dtype=tf.float32)),
        }

    # def call func
    def call(self, X, training=None, mask=None):
        """
        Forward layers in `VAE` to get the final result.
        :param X: (batch_size, n_x) - The original x.
        :param training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
        :param mask: A mask or list of masks. A mask can be either a tensor or None (no mask).
        :return X_reconstr: (batch_size, n_x) - The reconstructed x.
        :return loss: int - The whole loss, including reconstructed loss and latent loss.
        """
        # Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space.
        # z_mean - (batch_size, n_z)
        # z_log_sigma_sq - (batch_size, n_z)
        z_mean, z_log_sigma_sq = self._recognition(X)
        # Draw one sample z from Gaussian distribution, e.g. z = mu + sigma * epsilon
        # z - (batch_size, n_z)
        z = tf.add(z_mean, tf.multiply(
            tf.sqrt(tf.exp(z_log_sigma_sq)),
            tf.random.normal(z_log_sigma_sq.shape, mean=0., stddev=1., dtype=tf.float32)
        ))
        # Use generation network to determine mean of Bernoulli distribution of reconstructed x.
        # x_reconstr - (batch_size, n_x)
        x_reconstr = self._generation(z)
        # Calculate the whole loss, including reconstructed loss and latent loss.
        # loss - int
        loss = self._loss(X, x_reconstr, z_mean, z_log_sigma_sq)
        # Return the final X_reconstr & loss.
        return x_reconstr, loss

    # def recognite func
    def recognite(self, X):
        """
        Recognite data by mapping it into the latent space.
        :param X: (batch_size, n_x) - The original x.
        :return Z: (batch_size, n_z) - The transformed latent state.
        """
        # Note: This maps to the mean of distribution, we could alternatively sample from Gaussian distribution.
        # z_mean - (batch_size, n_z)
        z_mean, _ = self._recognition(X)
        # Return the final Z.
        return z_mean

    # def generate func
    def generate(self, Z=None):
        """
        Generate data by sampling from latent space. If Z is not None, data for this point in latent space
        is generated. Otherwise, Z is drawn from prior in latent space.
        :param Z: (batch_size, n_z) - The given latent state.
        :return X: (batch_size, n_x) - The generated x.
        """
        # If Z is None, draw Z from prior in latent space.
        Z = np.random.normal((self.params["n_z"],)) if Z is None else Z
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution.
        # x_reconstr_mean - (batch_size, n_x)
        x_reconstr_mean = self._generation(Z)
        # Return the final X.
        return x_reconstr_mean

    # def _recognition func
    def _recognition(self, x):
        """
        Generate probabilistic encoder (recognition network), which maps inputs onto a normal
        distribution in latent space. The transformation is parameterized and can be learned.
        :param x: (batch_size, n_x) - The original x.
        :return z_mean: (batch_size, n_z) - The mean of latent state.
        :return z_log_sigma_sq: (batch_size, n_z) - The log of squared sigma of latent state.
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
        # z_mean - (batch_size, n_z)
        z_mean = tf.add(
            tf.matmul(layer2, self.weights_["recognition"]["weights"]["out_mean"]),
            self.weights_["recognition"]["biases"]["out_mean"]
        )
        # z_log_sigma_sq - (batch_size, n_z)
        z_log_sigma_sq = tf.add(
            tf.matmul(layer2, self.weights_["recognition"]["weights"]["out_log_sigma"]),
            self.weights_["recognition"]["biases"]["out_log_sigma"]
        )
        # Return the final z_mean & z_log_sigma_sq.
        return z_mean, z_log_sigma_sq

    # def _generation func
    def _generation(self, z):
        """
        Generate probabilistic decoder (decoder network), which maps points in latent space onto
        a Bernoulli distribution in data space. The transformation is parameterized and can be learned.
        :param z: (batch_size, n_z) - The batch latent state.
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
        # x_reconstr_mean - (batch_size, n_x)
        x_reconstr_mean = tf.add(
            tf.matmul(layer2, self.weights_["generation"]["weights"]["out_mean"]),
            self.weights_["generation"]["biases"]["out_mean"]
        )
        # x_reconstr_log_sigma_sq - (batch_size, n_x)
        x_reconstr_log_sigma_sq = tf.add(
            tf.matmul(layer2, self.weights_["generation"]["weights"]["out_log_sigma"]),
            self.weights_["generation"]["biases"]["out_log_sigma"]
        )
        # Note: We direcly use x_reconstr_mean to represent x_recontr.
        # x_reconstr - (batch_size, n_x)
        x_reconstr = tf.nn.sigmoid(tf.where(True, x_reconstr_mean, tf.add(x_reconstr_mean, tf.multiply(
            tf.sqrt(tf.exp(x_reconstr_log_sigma_sq)),
            tf.random.normal(x_reconstr_log_sigma_sq.shape, mean=0., stddev=1., dtype=tf.float32)
        ))))
        # Return the final x_reconstr.
        return x_reconstr

    # def _loss func
    def _loss(self, x, x_reconstr, z_mean, z_log_sigma_sq):
        """
        Calculate the whole loss, including reconstruction loss and latent loss.
        :param x: (batch_size, n_x) - The original x.
        :param x_reconstr: (batch_size, n_x) - The mean of reconstructed x.
        :param z_mean: (batch_size, n_z) - The mean of latent state.
        :param z_log_sigma_sq: (batch_size, n_z) - The log of squared sigma of latent state.
        :return loss: int - The whole loss, including reconstructed loss and latent loss.
        """
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability of the input under the reconstructed Bernoulli
        #     distribution induced by the decoder in the data space). This can be interpreted as the number of "nats"
        #     required for reconstructing the input when the activation in latent is given.
        # Note: Adding 1e-10 to avoid evaluation of log(0, 0).
        # reconstr_loss - (batch_size,)
        reconstr_loss = -tf.reduce_sum(
            x * tf.math.log(1e-10 + x_reconstr) + (1. - x) * tf.math.log(1e-10 + 1. - x_reconstr), axis=-1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence between the distribution in latent
        #     space induced by the encoder on the data and some prior. This acts as a kind of regularizer. This can
        #     be interpreted as the number of "nats" required transmitting the latent space distribution given the prior.
        # latent_loss - (batch_size,)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), axis=-1)
        # Calculate the whole loss, averaging over batch.
        # loss - int
        loss = tf.reduce_mean(reconstr_loss + latent_loss)
        # Return the final loss.
        return loss

