#!/usr/bin/env python3
"""
Created on 13:48, Nov. 3rd, 2022

@author: Norbert Zheng
"""
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

# Prepare training process.
np.random.seed(22)
tf.random.set_seed(22)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
assert tf.__version__.startswith("2.")

# Get dataset from `tf.keras`.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Initialize image grid.
new_im = Image.new("L", (200, 200))
image_size = 28 * 28
h_dim = 512
z_dim = 20
num_epochs = 55
batch_size = 100
learning_rate = 1e-3

# def VAE class
class VAE(tf.keras.Model):
    """
    Variational Auto-Encoder.
    """

    def __init__(self):
        """
        Initialize VAE class.
        """
        # Initialize super to inherit `tf.keras.Model`-style class.
        super(VAE, self).__init__()

        ## Initialize variables.
        # input => h
        self.fc1 = tf.keras.layers.Dense(h_dim)
        # h => mu and variance
        self.fc2 = tf.keras.layers.Dense(z_dim)
        self.fc3 = tf.keras.layers.Dense(z_dim)
        # sampled z => h
        self.fc4 = tf.keras.layers.Dense(h_dim)
        # h => image
        self.fc5 = tf.keras.layers.Dense(image_size)

    # def encode func
    def encode(self, x):
        """
        Encode x to [mu,variance], shared MLP.
        """
        h = tf.nn.relu(self.fc1(x))
        # mu, log_variance
        return self.fc2(h), self.fc3(h)

    # def reparameterize func
    def reparameterize(self, mu, log_var):
        """
        Reparameterize trick.
        :param mu:
        :param log_var:
        :return:
        """
        std = tf.exp(log_var * 0.5)
        eps = tf.random.normal(std.shape)
        return mu + eps * std

    # def decode_logits func
    def decode_logits(self, z):
        """
        Decode logits, MLP.
        """
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)

    # def decode func
    def decode(self, z):
        """
        Normalize logits to get probability.
        """
        return tf.nn.sigmoid(self.decode_logits(z))

    # def call func
    def call(self, inputs, training=None, mask=None):
        """
        Forward VAE.
        """
        # Forward encoder.
        mu, log_var = self.encode(inputs)
        # Sample hidden state.
        z = self.reparameterize(mu, log_var)
        # Forward decoder.
        x_reconstructed_logits = self.decode_logits(z)
        # Return the final x_reconstructed_logits, mu, log_var.
        return x_reconstructed_logits, mu, log_var

# Instantiate VAE.
model = VAE()
model.build(input_shape=(4, image_size))
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Auto-Encoder does not need label.
dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(batch_size * 5).batch(batch_size)
num_batches = x_train.shape[0] // batch_size

# Create dir to save images.
dir_images = os.path.join(".", "images")
if not os.path.exists(dir_images): os.makedirs(dir_images)

# Training process.
for epoch in range(num_epochs):
    for step, x in enumerate(dataset):
        x = tf.reshape(x, (-1, image_size))
        with tf.GradientTape() as tape:
            # Forward pass.
            x_reconstruction_logits, mu, log_var = model(x)
            # Compute reconstruction loss and kl divergence.
            # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43.
            # Scaled by `image_size` for each individual pixel.
            reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_reconstruction_logits)
            reconstruction_loss = tf.reduce_sum(reconstruction_loss) / batch_size
            # Calculate KL divergence, for the mathematical details, please refer to
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kl_div = - 0.5 * tf.reduce_sum(1. + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
            kl_div = tf.reduce_mean(kl_div)
            # Calculate the whole loss.
            loss = tf.reduce_mean(reconstruction_loss) + kl_div
        # Backprop and optimize.
        gradients = tape.gradient(loss, model.trainable_variables)
        for g in gradients:
            tf.clip_by_norm(g, 15)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Log training information.
        if (step + 1) % 50 == 0:
            print((
                "Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
            ).format(epoch + 1, num_epochs, step + 1, num_batches, float(reconstruction_loss), float(kl_div)))
    # Generative model.
    z = tf.random.normal((batch_size, z_dim))
    # Decode with sigmoid.
    out = model.decode(z)
    out = tf.reshape(out, (-1, 28, 28)).numpy() * 255
    out = out.astype(np.uint8)
    # Since we cannot find image_grid function from version 2.0, we do it by hand.
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = out[index]
            im = Image.fromarray(im, mode="L")
            new_im.paste(im, (i,j))
            index += 1
    new_im.save(os.path.join(dir_images, "vae_sampled_epoch_{:d}.png".format(epoch + 1)))
    # Save the reconstructed images of last batch.
    out_logits, _, _ = model(x[:batch_size // 2])
    # out is just the logits, use sigmoid.
    out = tf.nn.sigmoid(out_logits)
    out = tf.reshape(out, (-1, 28, 28)).numpy() * 255
    # The original image.
    x = tf.reshape(x[:batch_size // 2], (-1, 28, 28))
    # Concat together.
    x_concat = tf.concat([x, out], axis=0).numpy() * 255
    x_concat = x_concat.astype(np.uint8)
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = x_concat[index]
            im = Image.fromarray(im, mode="L")
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(os.path.join(dir_images, "vae_reconstructed_epoch_{:d}.png".format(epoch + 1)))

