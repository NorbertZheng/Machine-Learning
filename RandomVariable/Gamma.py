#!/usr/bin/env python3
"""
Created on 19:58, Nov. 21st, 2022

@author: Norbert Zheng
"""
import numpy as np
from scipy.special import gamma
# local dep
from RandomVariable import RandomVariable

__all__ = [
    "Gamma",
]

# def Gamma class
class Gamma(RandomVariable):
    """
    Gamma distribution.
    >>> p(x|a,b) = b^a x^(a-1) exp(-bx) / gamma(a)
    """

    def __init__(self, a, b):
        """
        Initialize `Gamma` object.
        :param a: (n_x,) - The shape parameter of gamma distribution.
        :param b: (n_x,) - The rate parameter of gamma distribution.
        """
        # Initialize super to inherit `RandomVariable`-style class.
        super(Gamma, self).__init__()
        # Initialize parameters.
        a = np.asarray(a); b = np.asarray(b)
        assert a.shape == b.shape
        self.a = a; self.b = b

    """
    property funcs
    """
    @property
    def a(self):
        return self.params["a"]

    @a.setter
    def a(self, a):
        if isinstance(a, (int, float, np.number)):
            if a <= 0.:
                raise ValueError("ERROR: a must be positive in RandomVariable.Gamma.")
            self.params["a"] = np.asarray(a)
        elif isinstance(a, np.ndarray):
            if (a <= 0.).all():
                raise ValueError("ERROR: a must be positive in RandomVariable.Gamma.")
            self.params["a"] = a
        else:
            if a is not None:
                raise TypeError("ERROR: {} is not supported for a in RandomVariable.Gamma.".format(type(a)))
            self.params["a"] = None

    @property
    def b(self):
        return self.params["b"]

    @b.setter
    def b(self, b):
        if isinstance(b, (int, float, np.number)):
            if b <= 0.:
                raise ValueError("ERROR: b must be positive in RandomVariable.Gamma.")
            self.params["b"] = np.asarray(b)
        elif isinstance(b, np.ndarray):
            if (b <= 0.).any():
                raise ValueError("ERROR: b must be positive in RandomVariable.Gamma.")
            self.params["b"] = b
        else:
            if b is not None:
                raise TypeError("ERROR: {} is not supported for b in RandomVariable.Gamma.".format(type(b)))
            self.params["b"] = None

    @property
    def ndim(self):
        return self.a.ndim

    @property
    def size(self):
        return self.a.size

    @property
    def shape(self):
        return self.a.shape

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
        # Calculate the value of probability density function for each data point.
        # p - (n_data,)
        p = self.b**self.a\
            * X**(self.a - 1.)\
            * np.exp(-self.b * X)\
            / gamma(self.a)
        # Return the final p.
        return p

    # def _draw func
    def _draw(self, n_samples=1):
        """
        Draw samples from the distribution.
        :param n_samples: The number of samples to be sampled.
        :return samples: (n_samples, n_x) - The generated samples from the distribution.
        """
        # Draw samples from gamma distribution.
        # samples - (n_samples, n_mu)
        samples = np.random.gama(shape=self.a, scale=1/self.b, size=(n_samples,)+self.shape)
        # Return the final samples.
        return samples

