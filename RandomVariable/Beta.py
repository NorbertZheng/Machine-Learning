#!/usr/bin/env python3
"""
Created on 16:20, Nov. 21st, 2022

@author: Norbert Zheng
"""
import numpy as np
from scipy.special import gamma
# local dep
from RandomVariable import RandomVariable

__all__ = [
    "Beta",
]

# def Beta class
class Beta(RandomVariable):
    """
    Beta distribution.
    >>> p(mu|n_ones,n_zeros) = gamma(n_ones, n_zeros)\
    >>>                        * mu^(n_ones - 1) * (1 - mu)^(n_zeros - 1)\
    >>>                        / gamma(n_ones) / gamma(n_zeros)
    In the above formula, we use `gamma` function, which is often called the generalized factorial, e.g. `(x-1)!`.
    >>> \Gamma(z) = \int_{0}^{\infty} t^{z-1}e^{-t}dt.
    """

    def __init__(self, n_ones, n_zeros):
        """
        Initialize `Beta` object.
        :param n_ones: (n_mu,) - The pseudo-count of ones, e.g. alpha.
        :param n_zeros: (n_mu,) - The pseudo-count of zeros, e.g. beta.
        """
        # Initialize super to inherit `RandomVariable`-style class.
        super(Beta, self).__init__()
        # Initialize parameters.
        n_ones = np.asarray(n_ones); n_zeros = np.asarray(n_zeros)
        assert n_ones.shape == n_zeros.shape
        self.n_ones = n_ones; self.n_zeros = n_zeros

    """
    property funcs
    """
    @property
    def ndim(self):
        return self.n_ones.ndim

    @property
    def size(self):
        return self.n_ones.size

    @property
    def shape(self):
        return self.n_ones.shape

    """
    algo funcs
    """
    # def _pdf func
    def _pdf(self, mu):
        """
        Calculate the probability density function, e.g. `p(x;theta)` (or `p(x|theta)`).
        :param mu: (n_data, n_mu) - The observed data points.
        :return p: (n_data,) - The value of probability density function for each data point.
        """
        # Calculate the value of probability density function for each data point.
        # p - (n_data,)
        p = gamma(self.n_ones + self.n_zeros)\
            * np.power(mu, self.n_ones - 1) * np.power(1. - mu, self.n_zeros - 1)\
            / gamma(self.n_ones) / gamma(self.n_zeros)
        # Return the final p.
        return p

    # def _draw func
    def _draw(self, n_samples=1):
        """
        Draw samples from the distribution.
        :param n_samples: The number of samples to be sampled.
        :return samples: (n_samples, n_mu) - The generated samples from the distribution.
        """
        # Draw samples from beta distribution.
        # samples - (n_samples, n_mu)
        samples = np.random.beta(self.n_ones, self.n_zeros, size=(n_samples,)+self.shape)
        # Return the final samples.
        return samples

