#!/usr/bin/env python3
"""
Created on 19:15, Nov. 21st, 2022

@author: Norbert Zheng
"""
import numpy as np
from scipy.special import gamma
# local dep
from RandomVariable import RandomVariable

__all__ = [
    "Dirichlet",
]

# def Dirichlet class
class Dirichlet(RandomVariable):
    """
    Dirichlet distribution.
    >>> p(mu|alpha) = gamma(sum(alpha))\
    >>>               * prod_k mu_k ^ (alpha_k - 1)\
    >>>               / gamma(alpha_1) / ... / gamma(alpha_K)
    In the above formula, we use `gamma` function, which is often called the generalized factorial, e.g. `(x-1)!`.
    >>> \Gamma(z) = \int_{0}^{\infty} t^{z-1}e^{-t}dt.
    """

    def __init__(self, alpha):
        """
        Initialize `Dirichlet` object.
        :param alpha: (n_mu,) - The pseudo-count of each outcome, aka concentration parameter.
        """
        # Initialize super to inherit `RandomVariable`-style class.
        super(Dirichlet, self).__init__()
        # Initialize parameters.
        self.alpha = alpha

    """
    property funcs
    """
    @property
    def alpha(self):
        return self.params["alpha"]

    @alpha.setter
    def alpha(self, alpha):
        assert isinstance(alpha, np.ndarray)
        assert alpha.ndim == 1
        assert (alpha >= 0).all()
        self.params["alpha"] = alpha

    @property
    def ndim(self):
        return self.alpha.ndim

    @property
    def size(self):
        return self.alpha.size

    @property
    def shape(self):
        return self.alpha.shape

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
        p = gamma(np.sum(self.alpha))\
            * np.prod(mu ** (self.alpha - 1.), axis=-1)\
            / np.prod(gamma(self.alpha), axis=-1)
        # Return the final p.
        return p

    # def _draw func
    def _draw(self, n_samples=1):
        """
        Draw samples from the distribution.
        :param n_samples: The number of samples to be sampled.
        :return samples: (n_samples, n_mu) - The generated samples from the distribution.
        """
        # Draw samples from dirichlet distribution.
        # samples - (n_samples, n_mu)
        samples = np.random.dirichlet(self.alpha, size=(n_samples,)+self.shape)
        # Return the final samples.
        return samples

