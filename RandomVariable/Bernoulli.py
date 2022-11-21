#!/usr/bin/env python3
"""
Created on 16:17, Nov. 21st, 2022

@author: Norbert Zheng
"""
import numpy as np
# local dep
from RandomVariable import RandomVariable
from Beta import Beta

__all__ = [
    "Bernoulli",
]

# def Bernoulli class
class Bernoulli(RandomVariable):
    """
    Bernoulli distribution.
    >>> p(x|mu) = mu^x (1 - mu)^(1 - x)
    """

    def __init__(self, mu=None):
        """
        Initialize `Bernoulli` object.
        :param mu: (n_mu,) - The probability of value 1 for each element, could be `np.ndarray` or `Beta`.
        """
        # Initialize super to inherit `RandomVariable`-style class.
        super(Bernoulli, self).__init__()
        # Initialize parameters.
        self.mu = mu

    """
    property funcs
    """
    @property
    def mu(self):
        return self.params["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, (int, float, np.number)):
            if mu > 1. or mu < 0.:
                raise ValueError("ERROR: mu must be in [0, 1], not {:.2f} in RandomVariable.Bernoulli.".format(mu))
            self.params["mu"] = mu
        elif isinstance(mu, np.ndarray):
            if (mu > 1.).any() or (mu < 0.).any():
                raise ValueError("ERROR: mu must be in [0, 1], not {} in RandomVariable.Bernoulli.".format(mu))
            self.params["mu"] = mu
        elif isinstance(mu, Beta):
            self.params["mu"] = mu
        else:
            if mu is not None:
                raise TypeError("ERROR: {} is not supported for mu in RandomVariable.Bernoulli.".format(type(mu)))
            self.params["mu"] = None

    @property
    def ndim(self):
        return self.mu.ndim if hasattr(self.mu, "ndim") else None

    @property
    def size(self):
        return self.mu.size if hasattr(self.mu, "size") else None

    @property
    def shape(self):
        return self.mu.shape if hasattr(self.mu, "shape") else None

    """
    algo funcs
    """
    # def _fit func
    def _fit(self, X):
        """
        Estimate parameter(s) of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        # If `mu` is a random variable, which means that it
        # has prior, we use bayesian method to estimate it.
        if isinstance(self.mu, Beta):
            self._bayes(X)
        elif isinstance(self.mu, RandomVariable):
            raise NotImplementedError()
        # If `mu` is just a parameter, which means that it has no prior,
        # we just use max likelihood method to estimate it.
        else:
            self._ml(X)

    # def _ml func
    def _ml(self, X):
        """
        Use max likelihood method to estimate parameter(s) of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        # Get the pseudo-count.
        # n_ones - (n_mu,); n_zeros - (n_mu,)
        n_ones = np.sum(X == 1., axis=0); n_zeros = np.sum(X == 0., axis=0)
        assert X.size == np.sum(n_ones) + np.sum(n_zeros), ("ERROR: Input X must only has 0 or 1 in RandomVariable.Bernoulli.")
        # Set `mu` to the mean of `X`.
        self.mu = np.mean(X, axis=0)

    # def _map func
    def _map(self, X):
        """
        Use max a posterior method to estimate parameter(s) of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        assert isinstance(self.mu, Beta)
        assert X.shape[1:] == self.mu.shape
        # Calculate the evidence `p(X)`.
        # n_ones - (n_mu,); n_zeros - (n_mu,)
        n_ones = np.sum(X == 1., axis=0); n_zeros = np.sum(X == 0., axis=0)
        assert X.size == np.sum(n_ones) + np.sum(n_zeros), ("ERROR: Input X must only has 0 or 1 in RandomVariable.Bernoulli.")
        # Integrate evidence `p(X)` with likelihood `p(X|mu)` to get the posterior.
        n_ones += self.mu.n_ones; n_zeros += self.mu.n_zeros
        # Set `mu` to maximize posterior probability.
        self.mu = (n_ones - 1) / (n_ones + n_zeros - 2)

    # def _bayes func
    def _bayes(self, X):
        """
        Use bayesian method to estimate parameter(s) of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        assert isinstance(self.mu, Beta)
        assert X.shape[1:] == self.mu.shape
        # Calculate the evidence `p(X)`.
        # n_ones - (n_mu,); n_zeros - (n_mu,)
        n_ones = np.sum(X == 1., axis=0); n_zeros = np.sum(X == 0., axis=0)
        assert X.size == np.sum(n_ones) + np.sum(n_zeros), ("ERROR: Input X must only has 0 or 1 in RandomVariable.Bernoulli.")
        # Integrate evidence `p(X)` with likelihood `p(X|mu)` to get the posterior.
        self.mu.n_ones += n_ones; self.mu.n_zeros += n_zeros

    # def _pdf func
    def _pdf(self, X):
        """
        Calculate the probability density function, e.g. `p(x;theta)` (or `p(x|theta)`). Temporally not support calculating
        pdf when `mu` is `Beta`. Maybe we can just sample one, and then evaluate the corresponding pdf, just like in `_draw`.
        :param X: (n_data, n_x) - The observed data points.
        :return p: (n_data,) - The value of probability density function for each data point.
        """
        assert isinstance(X, np.ndarray)
        # Calculate the value of probability density function for each data point.
        # p - (n_data,)
        p = np.prod(self.mu**X * (1 - self.mu)**(1 - X))
        # Return the final p.
        return p

    # def _draw func
    def _draw(self, n_samples):
        """
        Draw samples from the distribution.
        :param n_samples: The number of samples to be sampled.
        :return samples: (n_samples, n_x) - The generated samples from the distribution.
        """
        # If `mu` is a parameter, we can directly use it to draw samples.
        if isinstance(self.mu, np.ndarray):
            samples = (self.mu > np.random.uniform(size=(n_samples,)+self.shape)).astype(np.int32)
        # If `mu` is a random variable following `Beta` distribution, use the mean to draw samples.
        elif isinstance(self.mu, Beta):
            samples = (self.mu.n_ones / (self.mu.n_ones + self.mu.n_zeros)\
                > np.random.uniform(size=(n_samples,)+self.shape)).astype(np.int32)
        # If `mu` is a random variable following unknown distribution, sample `mu` to draw samples.
        else:
            samples = self.mu.draw(n_samples) > np.random.uniform(size=(n_samples,)+self.shape).astype(np.int32)
        # Return the final samples.
        return samples

