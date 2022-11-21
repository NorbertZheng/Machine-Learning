#!/usr/bin/env python3
"""
Created on 20:13, Nov. 21st, 2022

@author: Norbert Zheng
"""
import numpy as np
# local dep
from RandomVariable import RandomVariable
from Gamma import Gamma

__all__ = [
    "Gaussian",
]

# def Gaussian class
class Gaussian(RandomVariable):
    """
    Gaussian distribution.
    >>> p(x|mu,var) = exp(-0.5 * (x - mu)^2 / var) / sqrt(2pi * var)
    """

    def __init__(self, mu=None, var=None, tau=None):
        """
        Initialize `Gaussian` object.
        :param mu: (n_x,) - The mean parameter of gaussian distribution.
        :param var: (n_x,) - The squared deviation parameter of gaussian distribution.
        :param tau: (n_x,) - The reciprocal of squared deviation parameter of gaussian distribution, e.g. `1/var`.
        """
        # Initialize super to inherit `RandomVariable`-style class.
        super(Gaussian, self).__init__()
        # Initialize parameters.
        # Note: `var` and `tau` are the same thing, we just need one parameter, and `var` is preferred.
        # And `if self.var is None: self.tau = tau` is not equal to `self.tau = tau if ... else None`!
        assert (var is not None) or (tau is not None)
        self.mu = mu; self.var = var
        if self.var is None: self.tau = tau

    """
    property funcs
    """
    @property
    def mu(self):
        return self.params["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, (int, float, np.number)):
            self.params["mu"] = np.array(mu)
        elif isinstance(mu, np.ndarray):
            self.params["mu"] = mu
        elif isinstance(mu, Gaussian):
            self.params["mu"] = mu
        else:
            if mu is not None:
                raise TypeError("ERROR: {} is not supported for mu in RandomVariable.Gaussian.".format(type(mu)))
            self.params["mu"] = None

    @property
    def var(self):
        return self.params["var"]

    @var.setter
    def var(self, var):
        if isinstance(var, (int, float, np.number)):
            assert var > 0.
            var = np.array(var)
            assert var.shape == self.shape
            self.params["var"] = var
            self.params["tau"] = 1 / var
        elif isinstance(var, np.ndarray):
            assert (var > 0.).all()
            assert tau.shape == self.shape
            self.params["var"] = var
            self.params["tau"] = 1 / var
        else:
            assert var is None
            self.params["var"] = None
            self.params["tau"] = None

    @property
    def tau(self):
        return self.params["tau"]

    @tau.setter
    def tau(self, tau):
        if isinstance(tau, (int, float, np.number)):
            assert tau > 0.
            tau = np.array(tau)
            assert tau.shape == self.shape
            self.params["tau"] = tau
            self.params["var"] = 1 / tau
        elif isinstance(tau, np.ndarray):
            assert (tau > 0.).all()
            assert tau.shape == self.shape
            self.params["tau"] = tau
            self.params["var"] = 1 / tau
        elif isinstance(tau, Gamma):
            assert tau.shape == self.shape
            self.params["tau"] = tau
            self.params["var"] = None
        else:
            assert tau is None
            self.params["tau"] = None
            self.params["var"] = None

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
        mu_is_gaussian = isinstance(self.mu, Gaussian)
        tau_is_gamma = isinstance(self.tau, Gamma)
        # If both `mu` and `tau` are random variables, raise `NotImplementedError`.
        if mu_is_gaussian and tau_is_gamma:
            raise NotImplementedError()
        # If `mu` is a random variable, which means that it has prior, we use bayesian method to estimate it.
        elif mu_is_gaussian:
            self._bayes_mu(X)
        # If `tau` is a random variable, which means that it has prior, we use bayesian method to estimate it.
        elif tau_is_gamma:
            self._bayes_tau(X)
        # If both `mu` and `tau` are just parameters, which means that it has no prior,
        # we just use max likelihood method to estimate it.
        else:
            self._ml(X)

    # def _ml func
    def _ml(self, X):
        """
        Use max likelihood method to estimate parameter(s) of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        # Set `mu` to the mean of `X`, and `var` to the variance of `X`.
        self.mu = np.mean(X, axis=0); self.var = np.var(X, axis=0)

    # def _map func
    def _map(self, X):
        """
        Use max a posterior method to estimate `mu` of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        assert isinstance(self.mu, Gaussian)
        assert isinstance(self.var, np.ndarray)
        # Calculate `mu` from the evidence `p(x)`.
        # mu - (n_x,)
        mu = np.mean(X, axis=0)
        # Set `mu` to maximize posterior probability.
        self.mu = (self.tau * self.mu.mu + len(X) * self.mu.tau * mu) / (len(X) * self.mu.tau + self.tau)

    # def _bayes_mu func
    def _bayes_mu(self, X):
        """
        Use bayesian method to estimate `mu` of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        # Calculate `mu` from the evidence `p(x)`.
        # mu - (n_x,); tau - (n_x,)
        mu = np.mean(X, axis=0); tau = self.mu.tau + len(X) * self.tau
        # Integrate evidence `p(X)` with likelihood `p(X|mu)` to get the posterior.
        self.mu = Gaussian(
            mu=(self.mu.mu * self.mu.tau + len(X) * mu * self.tau) / tau,
            tau=tau
        )

    # def _bayes_tau func
    def _bayes_tau(self, X):
        """
        Use bayesian method to estimate `tau` of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        # Calculate `var` from the evidence `p(x)`.
        # var - (n_x,)
        var = np.var(X, axis=0)
        # Integrate evidence `p(X)` with likelihood `p(X|mu)` to get the posterior.
        self.tau = Gamma(
            a=self.tau.a + 0.5 * len(X),
            b=self.tau.b + 0.5 * len(X) * var
        )

    # def _bayes func
    def _bayes(self, X):
        """
        Use bayesian method to estimate parameter(s) of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        mu_is_gaussian = isinstance(self.mu, Gaussian)
        tau_is_gamma = isinstance(self.tau, Gamma)
        # If only `mu` is a random variable, this is the same with `_bayes_mu`.
        if mu_is_gaussian and not tau_is_gamma:
            mu = np.mean(X, axis=0); tau = self.mu.tau + len(X) * self.tau
            self.mu = Gaussian(
                mu=(self.mu.mu * self.mu.tau + len(X) * mu * self.tau) / tau,
                tau=tau
            )
        # If only `tau` is a random variable, this is the same with `_bayes_tau`.
        elif not mu_is_gaussian and tau_is_gamma:
            var = np.var(X, axis=0)
            self.tau = Gamma(
                a=self.tau.a + 0.5 * len(X),
                b=self.tau.b + 0.5 * len(X) * var
            )
        # If both `mu` and `tau` are random variables.
        elif mu_is_gaussian and tau_is_gamma:
            raise NotImplementedError()
        # Neither `mu` nor `tau` is a random variable, should use max likelihood method.
        else:
            raise NotImplementedError()

    # def _pdf func
    def _pdf(self, X):
        """
        Calculate the probability density function, e.g. `p(x;theta)` (or `p(x|theta)`). Temporally not support calculating
        pdf when `mu` is `Beta`. Maybe we can just sample one, and then evaluate the corresponding pdf, just like in `_draw`.
        :param X: (n_data, n_x) - The observed data points.
        :return p: (n_data,) - The value of probability density function for each data point.
        """
        # Calculate the value of probability density function for each data point.
        # p - (n_data,)
        p = np.exp(-0.5 * self.tau * (X - self.mu)**2) / np.sqrt(2 * np.pi * self.var)
        # Return the final p.
        return p

    # def _draw func
    def _draw(self, n_samples=1):
        """
        Draw samples from the distribution.
        :param n_samples: The number of samples to be sampled.
        :return samples: (n_samples, n_x) - The generated samples from the distribution.
        """
        # Draw samples from normal distribution.
        # samples - (n_samples, n_mu)
        samples = np.random.normal(loc=self.mu, scale=np.sqrt(self.var), size=(n_samples,)+self.shape)
        # Return the final samples.
        return samples

