#!/usr/bin/env python3
"""
Created on 16:04, Nov. 10th, 2022

@author: Norbert Zheng
"""
import time
import copy as cp
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
# local dep
import configs, params, models, utils

# Load data. We should note that the diagonal of adj is 0s.
adj, features = utils.load_data(configs.args.dataset)
adj, features = adj.astype(np.float32), features.astype(np.float32)
assert np.diag(adj.toarray()).sum() == 0
print(adj.shape, features.shape)

# Initialize params.
params_inst = cp.deepcopy(getattr(params, "_".join([configs.args.model, "params"])))

# Use adj to prepare [train,val,test] set.
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = utils.prepare_train(adj)
print(adj_train.shape, train_edges.shape, val_edges.shape,
    val_edges_false.shape, test_edges.shape, test_edges_false.shape)
# adj - (n_nodes, n_nodes)
adj = adj_train
# Calculate the adjacency matrix.
adjacency = [
    # The original adjacency matrix with diagonal as 0s.
    utils.sparse2tuple(adj),
    # The target adjacency matrix with diagonal as 1s.
    utils.sparse2tuple(adj + sp.eye(adj.shape[0], dtype=np.float32)),
    # The normalized adjacency matrix with disgonal as >0s.
    utils.sparse2tuple(utils.preprocess_adj(adj))
]

# Re-build features if not use_feature.
# features - (n_nodes, n_x) if use_feature else (n_nodes, n_nodes)
if not params_inst["use_feature"]: features = sp.identity(features.shape[0])
features = utils.sparse2tuple(features.tocoo())
params_inst["n_x"] = features[2][1]
params_inst["n_x_sparse"] = features[1].shape[0]
features = tf.SparseTensor(*features)

# Create model.
model = getattr(models, configs.args.model.upper())(params_inst, adjacency)

# Initialize optimizer related variables.
optimizer = tf.keras.optimizers.Adam(lr=params_inst["learning_rate"])

# def get_roc_score func
def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        emb = model.encode(features, dropout=0.).numpy()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

# Get `tf.Tensor` of adj_label.
# adj_label - (n_nodes, n_nodes)
adj_label = tf.reshape(tf.cast(utils.tuple2sparse(adjacency[1]).todense(), dtype=tf.float32), (-1,))
# Training process.
val_roc_score = []
for epoch_idx in range(params_inst["n_epochs"]):
    # Record the start time.
    time_start = time.time()
    # Use one-epoch data to forward model.
    with tf.GradientTape() as tape:
        adj_reconstr, loss = model(features, dropout=params_inst["dropout"])
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Calculate accuracy using reconstructed adjacency matrix.
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.greater_equal(
        tf.sigmoid(adj_reconstr), 0.5), dtype=tf.int32), tf.cast(adj_label, dtype=tf.int32)), dtype=tf.float32))
    # Calculate roc related metrics.
    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)
    # Print log information.
    print("Epoch:{:4d}".format(epoch_idx + 1), "train_loss={:.5f}".format(loss.numpy()),
          "train_acc=", "{:.5f}".format(accuracy.numpy()), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr), "time=", "{:.5f}".format(time.time() - time_start))

print("Optimization Finished!")

roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
print("Test ROC score: " + str(roc_score))
print("Test AP score: " + str(ap_score))

