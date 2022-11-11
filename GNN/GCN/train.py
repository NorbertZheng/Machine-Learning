#!/usr/bin/env python3
"""
Created on 16:45, Oct. 31st, 2022

@author: Norbert Zheng
"""
import os, time
import numpy as np
import tensorflow as tf
# local dep
import utils, configs, models

# Define training macro.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print("tf version:", tf.__version__)
assert tf.__version__.startswith("2.")

# Set random seed.
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# Load data.
print(configs.args.dataset)
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = utils.load_data(configs.args.dataset)
print("adj:", adj.shape)
print("features:", features.shape)
print("y:", y_train.shape, y_val.shape, y_test.shape)
print("mask:", train_mask.shape, val_mask.shape, test_mask.shape)

# D^-1@X
# [49216, 2], [49216], [2708, 1433]
features = utils.preprocess_features(features)
print("features coordinates::", features[0].shape)
print("features data::", features[1].shape)
print("features shape::", features[2])

if configs.args.model == "gcn":
    # D^-0.5 A D^-0.5
    support = [utils.preprocess_adj(adj)]
    num_supports = 1
    model_func = models.GCN
elif configs.args.model == "gcn_cheby":
    support = utils.chebyshev_polynomials(adj, configs.args.max_degree)
    num_supports = 1 + configs.args.max_degree
    model_func = models.GCN
elif configs.args.model == "dense":
    support = [utils.preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = models.MLP
else:
    raise ValueError("Invalid argument for model: " + str(configs.args.model))

# Create model.
model = models.GCN(input_dim=features[2][1], output_dim=y_train.shape[1], num_features_nonzero=features[1].shape)

train_label = tf.convert_to_tensor(y_train)
train_mask = tf.convert_to_tensor(train_mask)
val_label = tf.convert_to_tensor(y_val)
val_mask = tf.convert_to_tensor(val_mask)
test_label = tf.convert_to_tensor(y_test)
test_mask = tf.convert_to_tensor(test_mask)
features = tf.SparseTensor(*features)
support = [tf.cast(tf.SparseTensor(*support[0]), dtype=tf.float32)]
num_features_nonzero = features.values.shape
dropout = configs.args.dropout

optimizer = tf.keras.optimizers.Adam(lr=1e-2)

for epoch in range(configs.args.epochs):
    with tf.GradientTape() as tape:
        loss, acc = model((features, train_label, train_mask, support))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    _, val_acc = model((features, val_label, val_mask, support), training=False)
    if epoch % 20 == 0: print(epoch, float(loss), float(acc), "\tval:", float(val_acc))

test_loss, test_acc = model((features, test_label, test_mask, support), training=False)
print("\ttest:", float(test_loss), float(test_acc))

