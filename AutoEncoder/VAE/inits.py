#!/usr/bin/env python3
"""
Created on 20:10, Nov. 5th, 2022

@author: Norbert Zheng
"""
import numpy as np
import tensorflow as tf

__all__ = [
    "xavier",
]

# def xavier func
def xavier(shape, scale=1., name=None):
    """
    Xavier initializer, also called Glorot & Bengio (AISTATS 2010) initialization.
    :param shape: (2,) - The shape of weight mnatrix to be initialized.
    :param scale: int - The scale constant of Xavier initialization.
    :param name: str - The name of weight matrix to be initialized.
    :return initial: (shape[0], shape[1]) - The xavier-initialized weight matrix.
    """
    init_range = scale * np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

