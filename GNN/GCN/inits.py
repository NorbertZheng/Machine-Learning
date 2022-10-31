#!/usr/bin/env python3
"""
Created on 11:31, Oct. 31st, 2022

@author: Norbert Zheng
"""
import numpy as np
import tensorflow as tf

__all__ = [
    "uniform",
    "glorot",
    "zeros",
    "ones",
]

# def uniform func
def uniform(shape, scale=0.05, name=None):
    """
    Uniform initialization.
    """
    initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# def glorot func
def glorot(shape, name=None):
    """
    Glorot & Bengio (AISTATS 2010) initialization.
    """
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# def zeros func
def zeros(shape, name=None):
    """
    All zeros initialization.
    """
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# def ones func
def ones(shape, name=None):
    """
    All ones initialization.
    """
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

if __name__ == "__main__":
    uniform_inst = uniform((5,4))
    glorot_inst = glorot((5,4))
    zeros_inst = zeros((5,4))
    ones_inst = zeros((5,4))

