#!/usr/bin/env python3
"""
Created on 22:05, Nov. 5th, 2022

@author: Norbert Zheng
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# local dep
import params, models

# Initialize random seed.
np.random.seed(0)
tf.random.set_seed(0)

# Initialize dir_images.
dir_images = os.path.join(".", "images")
if not os.path.exists(dir_images): os.makedirs(dir_images)

# Load MNIST data in a format suited for tensorflow.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32)
x_test = x_test.reshape((x_test.shape[0], -1)).astype(np.float32)
x_train /= 255.; x_test /= 255.
# Initialize params, and then instantiate VAE.
params_inst = params.default_params
vae_inst = models.VAE(params_inst)
# Initialize optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=params_inst["learning_rate"])

## Training process.
for epoch_idx in range(params_inst["n_epochs"]):
    # Initialize loss.
    loss = 0.
    # Loop over all batches.
    for batch_idx in range(x_train.shape[0] // params_inst["batch_size"]):
        # Get X of current training batch.
        # X_i - (batch_size, n_x)
        X_i = x_train[batch_idx*params_inst["batch_size"]:(batch_idx+1)*params_inst["batch_size"],:]
        # Train the vae_inst using one-batch data.
        with tf.GradientTape() as gt:
            _, loss_i = vae_inst(X_i, training=True)
        # Modify weights to optimize the model.
        gradients_i = gt.gradient(loss_i, vae_inst.trainable_variables)
        gradients_i = [(tf.clip_by_norm(grad, 2), var)\
            if grad is not None else (grad, var)\
            for (grad, var) in zip(gradients_i, vae_inst.trainable_variables)]
        optimizer.apply_gradients(gradients_i)
        # Calculate average loss.
        loss += loss_i / (x_train.shape[0] // params_inst["batch_size"])
    # Display logs per n_log step.
    if epoch_idx % params_inst["n_log"] == 0:
        print("Epoch: {:04d}\tLoss: {:.9f}".format(epoch_idx, loss))

## Reconstruction process.
# Sample x from x_test.
# x_sample - (n_samples_reconstr, n_x)
x_sample = x_test[:params_inst["n_samples_reconstr"],:]
# Reconstruct x from x_sample.
# x_reconstr - (n_samples_reconstr, n_x)
x_reconstr, _ = vae_inst(x_sample)
x_reconstr = x_reconstr.numpy()
# Plot sample & reconstructed images.
plt.figure(figsize=(8, 12))
for sample_idx in range(params_inst["n_samples_reconstr"]):
    plt.subplot(5, 2, 2 * sample_idx + 1)
    plt.imshow(x_sample[sample_idx].reshape(28, 28), vmin=0., vmax=1., cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2 * sample_idx + 2)
    plt.imshow(x_reconstr[sample_idx].reshape(28, 28), vmin=0., vmax=1., cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(dir_images, "reconstruction.png"))
plt.close("all")

## Recognition process.
# Sample x from x_test, y from y_test.
# x_sample - (n_samples_recog, n_x)
# y_sample - (n_samples_recog,)
x_sample, y_sample = x_test[:params_inst["n_samples_recog"],:], y_test[:params_inst["n_samples_recog"]]
# Recognite z_sample from x_sample.
# z_sample - (n_samples_recog, n_z)
z_sample = vae_inst.recognite(x_sample)
# Plot the structure of the learned manifold.
plt.figure(figsize=(8, 6)) 
plt.scatter(z_sample[:,0], z_sample[:, 1], c=y_sample)
plt.colorbar()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(dir_images, "recognition.png"))
plt.close("all")

## Generation process.
nx = ny = 20
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)
canvas = np.empty((28*ny, 28*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_sample_i = np.array([[xi, yi]]*params_inst["batch_size"], dtype=np.float32)
        x_sample_i = vae_inst.generate(z_sample_i).numpy()
        canvas[(nx-i-1)*28:(nx-i)*28,j*28:(j+1)*28] = x_sample_i[0].reshape(28, 28)
# Plot reconstrunctions at the positions in the latent space.
plt.figure(figsize=(8, 10))        
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.tight_layout()
plt.savefig(os.path.join(dir_images, "generation.png"))
plt.close("all")

