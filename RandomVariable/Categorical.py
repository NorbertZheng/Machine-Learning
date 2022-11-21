#!/usr/bin/env python3
"""
Created on 19:29, Nov. 21st, 2022

@author: Norbert Zheng
"""
import numpy as np
# local dep
from RandomVariable import RandomVariable
from Dirichlet import Dirichlet

__all__ = [
    "Categorical",
]

# def Categorical class
class Categorical(RandomVariable):
    """
    Categorical distribution.
    >>> p(x|mu) = prod_k mu_k^x_k
    """

    def __init__(self, mu=None):
        """
        Initialize `Categorical` object.
        :param mu: (n_classes,) - The probability of each class, could be `np.ndarray` or `Dirichlet`.
        """
        # Initialize super to inherit `RandomVariable`-style class.
        super(Categorical, self).__init__()
        # Initialize parameters.
        self.mu = mu

    """
    property funcs
    """
    @property
    def mu(self):
        return self.params["mu"]

    @mu.setter
    def mu(self):
        if isinstance(mu, np.ndarray):
            if mu.ndim != 1:
                raise ValueError("ERROR: The dimensionality of mu must be 1 in RandomVariable.Categorical.")
            if (mu < 0.).any():
                raise ValueError("ERROR: mu must be non-negative in RandomVariable.Categorical.")
            if not np.allclose(np.sum(mu), 1.):
                raise ValueError("ERROR: The sum of mu must be 1 in RandomVariable.Categorical.")
            self.n_classes = mu.size
            self.params["mu"] = mu
        elif isinstance(mu, Dirichlet):
            self.n_classes = mu.size
            self.params["mu"] = mu
        else:
            if mu is not None:
                raise TypeError("ERROR: {} is not supported for mu in RandomVariable.Categorical.".format(type(mu)))
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
    internal funcs
    """
    # def _check_input func
    def _check_input(self, X):
        """
        Check whether input has the specified properties.
        :param X: (n_data, n_x) - The observed data points.
        """
        assert X.ndim == 2
        assert (X >= 0.).all()
        assert (np.sum(X, axis=-1) == 1.).all()

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
        if isinstance(self.mu, Dirichlet):
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
        # Check whether input has the specified properties.
        self._check_input(X)
        # Set `mu` to the mean of `X`.
        self.mu = np.mean(X, axis=0)

    # def _map func
    def _map(self, X):
        """
        Use max a posterior method to estimate parameter(s) of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        # Check whether input has the specified properties.
        self._check_input(X)
        assert isinstance(self.mu, Dirichlet)
        # Integrate evidence `p(X)` with likelihood `p(X|mu)` to get the posterior.
        alpha = np.sum(X, axis=0) + self.mu.alpha
        # Set `mu` to maximize the posterior probability.
        self.mu = (alpha - 1.) / np.sum(alpha - 1.)

    # def _bayes func
    def _bayes(self, X):
        """
        Use bayesian method to estimate parameter(s) of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        # Check whether input has the specified properties.
        self._check_input(X)
        assert isinstance(self.mu, Dirichlet)
        # Integrate evidence `p(X)` with likelihood `p(X|mu)` to get the posterior.
        self.mu.alpha += np.sum(X, axis=0)

    # def _pdf func
    def _pdf(self, X):
        """
        Calculate the probability density function, e.g. `p(x;theta)` (or `p(x|theta)`). Temporally not support calculating
        pdf when `mu` is `Beta`. Maybe we can just sample one, and then evaluate the corresponding pdf.
        :param X: (n_data, n_x) - The observed data points.
        :return p: (n_data,) - The value of probability density function for each data point.
        """
        # Check whether input has the specified properties.
        self._check_input(X)
        assert isinstance(self.mu, np.ndarray)
        # Calculate the value of probability density function for each data point.
        # p - (n_data,)
        p = np.prod(self.mu ** X, axis=-1)
        # Return the final p.
        return p

    # def _draw func
    def _draw(self, n_samples=1):
        """
        Draw samples from the distribution. Temporally not support calculating pdf when `mu` is `Beta`.
        Maybe we can just sample one, and then evaluate the corresponding pdf.
        :param n_samples: The number of samples to be sampled.
        :return samples: (n_samples, n_x) - The generated samples from the distribution.
        """
        assert isinstance(self.mu, np.ndarray)
        # Draw samples from choice distribution.
        # samples - (n_samples, n_x)
        samples = np.eye(self.n_classes)[np.random.choice(self.n_classes, n_samples, p=self.mu)]
        # Return the final samples.
        return samples

