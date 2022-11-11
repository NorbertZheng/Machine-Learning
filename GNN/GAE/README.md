# Graph Auto-Encoders

This is a TensorFlow implementation of the (Variational) Graph Auto-Encoder model as described in our paper:

 - T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016).

Graph Auto-Encoders (GAEs) are end-to-end trainable neural network models for unsupervised learning, clustering and link prediction on graphs.

![](./graph-variational-autoencoder.png)

GAEs have successfully been used for:
 - Link prediction in large-scale relational data: M. Schlichtkrull & T. N. Kipf et al., [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (2017).
 - Matrix completion / recommendation with side information: R. Berg et al., [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) (2017).

GAEs are based on Graph Convolutional Networks (GCNs), a recent class of models for end-to-end (semi-)supervised learning on graphs:
 - T. N. Kipf, M. Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR (2017).

A high-level introduction is given in our blog post:
 - Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016).

## Requirements
* tensorflow (2.4.0)
* networkx
* scikit-learn
* scipy

## Run the demo

```bash
python train.py
```

## Data

Demo data can be found [here](https://github.com/tkipf/gae/tree/master/gae/data).

In order to use your own data, you have to provide 
* (required) an N by N adjacency matrix (N is the number of nodes),
* (optional) an N by D feature matrix (D is the number of features per node).

Have a look at the `load_data()` function in `utils.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found [here](http://linqs.cs.umd.edu/projects/projects/lbc/). And its different format can be found [here](https://github.com/kimiyoung/planetoid). 

You can specify a dataset as follows:

```bash
python train.py --dataset citeseer
```

(or by editing `train.py`)

## Models

You can choose between the following models: 
* `gae`: Graph Auto-Encoder (with GCN encoder).
* `gvae`: Variational Graph Auto-Encoder (with GCN encoder).

```bash
python train.py --model gvae
```

## Cite

Please cite our paper if you use this code in your own work:

```
@article{kipf2016variational,
  title={Variational Graph Auto-Encoders},
  author={Kipf, Thomas N and Welling, Max},
  journal={NIPS Workshop on Bayesian Deep Learning},
  year={2016}
}
```

