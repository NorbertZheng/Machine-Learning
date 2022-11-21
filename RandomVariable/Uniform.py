#!/usr/bin/env python3
"""
Created on 15:37, Nov. 21st, 2022

@author: Norbert Zheng
"""
import numpy as np
# local dep
from RandomVariable import RandomVariable

__all__ = [
    "Uniform",
]

# def Uniform class
class Uniform(RandomVariable):
    """
    Uniform distribution.
    >>> p(x|a,b) = 1 / ((b0 - a0) * (b1 - a1)), if a <= x <= b,
    >>>          = 0, else
    """

    def __init__(self, low, high):
        """
        Initialize `Uniform` object.
        :param low: (n_x,) - The lower boundary of data points, could be `int`, `float`, or `np.ndarray`.
        :param high: (n_x,) - The higher boundary of data points, could be `int`, `float`, or `np.ndarray`.
        """
        # Initialize super to inherit `RandomVariable`-style class.
        super(Uniform, self).__init__()
        # Initialize parameters.
        low = np.asarray(low); high = np.asarray(high)
        assert low.shape == high.shape
        assert (low <= high).all()
        self.low = low; self.high = high
        self.value = 1 / np.prod(high - low)

    """
    property funcs
    """
    @property
    def low(self):
        return self.params["low"]

    @low.setter
    def low(self, low):
        self.params["low"] = low

    @property
    def high(self):
        return self.params["high"]

    @high.setter
    def high(self, high):
        self.params["high"] = high

    @property
    def ndim(self):
        return self.low.ndim

    @property
    def size(self):
        return self.low.size

    @property
    def shape(self):
        return self.low.shape

    @property
    def mean(self):
        return 0.5 * (self.low + self.high)

    """
    algo funcs
    """
    # def _pdf func
    def _pdf(self, X):
        """
        Calculate the probability density function, e.g. `p(x;theta)` (or `p(x|theta)`).
        :param X: (n_data, n_x) - The observed data points.
        :return p: (n_data,) - The value of probability density function for each data point.
        """
        # Limit the input data points to the specified range.
        # higher - (n_data,); lower - (n_data,)
        higher = np.logical_and.reduce(X >= self.low, axis=-1)
        lower = np.logical_and.reduce(X <= self.high, axis=-1)
        # Calculate the value of probability density function for each data point.
        # p - (n_data,)
        p = self.value * np.logical_and(higher, lower)
        # Return the final p.
        return p

    # def _draw func
    def _draw(self, n_samples=1):
        """
        Draw samples from the distribution.
        :param n_samples: The number of samples to be sampled.
        :return samples: (n_samples, n_x) - The generated samples from the distribution.
        """
        # Draw samples from [0,1]-uniform distribution.
        # u01 - (n_samples, n_x)
        u01 = np.random.uniform(size=(n_samples,)+self.shape)
        # Use transformation method to get the corresponding value.
        # samples - (n_samples, n_x)
        samples = u01 * (self.high - self.low) + self.low
        # Return the final samples.
        return samples

