#!/usr/bin/env python3
"""
Created on 14:46, Nov. 21st, 2022

@author: Norbert Zheng
"""
import numpy as np

__all__ = [
    "RandomVariable",
]

# def RandomVariable class
class RandomVariable(object):
    """
    Base class for random variables.
    """

    def __init__(self):
        """
        Initialize `RandomVariable` object.
        """
        self.params = {}

    def __repr__(self):
        """
        Modify display information.
        :return info: str - The display information.
        """
        # Set display information.
        info = "{}(\n".format(self.__class__.__name__)
        for key_i, value_i in self.params.items():
            info += (" " * 4)
            if isinstance(value_i, RandomVariable):
                info += "{}={:8}".format(key_i, value_i)
            else:
                info += "{}={}".format(key_i, value_i)
            info += "\n"
        info += ")"
        # Return the final info.
        return info

    def __format__(self, indent="4"):
        """
        Modify formatted information.
        :param indent: str - The indent of the format information.
        :rerturn info: str - The formatted information.
        """
        # Since only str can be passed into `__format__` function, we convert indent to integer.
        indent = int(indent)
        # Set formatted information.
        info = "{}(\n".format(self.__class__.__name__)
        for key_i, value_i in self.params.items():
            info += (" " * indent)
            if isinstance(value_i, RandomVariable):
                info += "{}=".format(key_i) + value_i.__format__(str(indent + 4))
            else:
                info += "{}={}".format(key_i, value_i)
            info += "\n"
        info += (" " * (indent - 4)) + ")"
        # Return the final info.
        return info

    """
    algo funcs
    """
    # def fit func
    def fit(self, X, **kwargs):
        """
        Estimate parameter(s) of the distribution.
        :param X: (n_data, n_x) - The observed data points.
        """
        # Check whether input has the specified properties.
        self._check_input(X)
        # Fit data to get the parameter(s) of the distribution.
        if hasattr(self, "_fit"):
            self._fit(X, **kwargs)
        else:
            raise NotImplementedError()

    # def pdf func
    def pdf(self, X):
        """
        Calculate the probability density function, e.g. `p(x;theta)` (or `p(x|theta)`).
        :param X: (n_data, n_x) - The observed data points.
        :return p: (n_data,) - The value of probability density function for each data point.
        """
        # Check whether input has the specified properties.
        self._check_input(X)
        # Calculate the probability density function for each data point.
        if hasattr(self, "_pdf"):
            return self._pdf(X)
        else:
            raise NotImplementedError()

    # def draw func
    def draw(self, n_samples=1):
        """
        Draw samples from the distribution.
        :param n_samples: The number of samples to be sampled.
        :return samples: (n_samples, n_x) - The generated samples from the distribution.
        """
        assert isinstance(n_samples, int)
        # Draw samples from the distribution.
        if hasattr(self, "_draw"):
            return self._draw(n_samples=n_samples)
        else:
            raise NotImplementedError()

    """
    internal funcs
    """
    # def _check_input func
    def _check_input(self, X):
        """
        Check whether input has the specified properties.
        :param X: (n_data, n_x) - The observed data points.
        """
        assert isinstance(X, np.ndarray)

