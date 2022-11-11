#!/usr/bin/env python3
"""
Created on 21:39, Nov. 9th, 2022

@author: Norbert Zheng
"""
import numpy as np
import tensorflow as tf

__all__ = [
    "glorot",
    "zeros",
]

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

