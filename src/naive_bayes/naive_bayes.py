from abc import abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy

from .._base.base_model import BaseModel
from .._utils.numerical import laplace_smoothing
from .._utils.checkers import check_input_shape


class NaiveBayes(BaseModel):
    """
    Classifier; cannot be used as a regressor since output has to be a posterior probability,
    and is generally not good at predicting exact probabilities.
    """
    def __init__(self):
        self._params = defaultdict(dict)

    @check_input_shape
    def fit(self, X, y):
        self._check_input(X)
        for label in y.unique():
            self._params[label]["prior"] = self._calc_prior(y, label)
            self._params[label]["likelihood_func"] = self._set_likelihood_func(X, y, label)

    def _calc_prior(self, y, label):
        """
        Can prompt the user to pass in a prior probability, but for now we just calculate it.
        """
        return (y == label).mean()

    @abstractmethod
    def _set_likelihood_func(self, X, y, label):
        """
        Bernoulli, multinomial, Gaussian

        And of course, we assume conditional independence between the features,
        given the label (naive independence assumption) in all cases.

        Must return a function that takes in a vector of shape (n_samples, n_features)
        and returns a vector (pd.Series) of shape (n_samples,).
        """
        pass

    @check_input_shape
    def predict(self, X):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet.")

        self._check_input(X)

        pred = {}
        for label in self._params.keys():
            prior = self._params[label]["prior"]
            likelihood_func = self._params[label]["likelihood_func"]
            posterior = prior * likelihood_func(X) # numerator only, to be exact
            pred[label] = posterior

        return pd.concat(pred, axis=1).idxmax(axis=1)


class BernoulliNaiveBayes(NaiveBayes):
    """
    Input, X, is a binary matrix of shape
    (n_samples, n_features), where each feature is either 0 or 1.
    The features are assumed to be Bernoulli distributed.
    """
    def _set_likelihood_func(self, X, y, label):
        num = X[y == label].sum()
        denom = len(X[y == label])
        positive_proba = laplace_smoothing(num, denom, alpha=1)

        def output(X):
            probas = np.where(X == 1, positive_proba, 1 - positive_proba)
            return pd.Series(probas.prod(axis=1), index=X.index)

        return output

    def _check_input(self, X):
        is_one = X == 1
        is_zero = X == 0
        if not (is_one | is_zero).all().all().all():
            raise ValueError("Input must be a binary vector consisting of only 0s and 1s.")


class MultinomialNaiveBayes(NaiveBayes):
    """
    Input, X, is a inteteger matrix of shape
    (n_samples, n_features), where each feature is a count of occurrences of something.
    The features are assumed to be multinomially distributed.
    """
    def _set_likelihood_func(self, X, y, label):
        num = X[y == label].sum()
        denom = num.sum()
        frequency_proba = laplace_smoothing(num, denom, alpha=1)

        def output(X):
            probas = frequency_proba ** X
            return pd.Series(probas.prod(axis=1), index=X.index)
        
        return output

    def _check_input(self, X):
        if not all([pd.api.types.is_integer_dtype(X[col]) for col in X.columns]): # can only check for each column
            raise TypeError("Input must be an integer matrix.")
        if (X < 0).any().any():
            raise ValueError("Input must be a non-negative integer matrix.")


class GaussianNaiveBayes(NaiveBayes):
    """
    Input is some normally distributed variable
    """
    def _set_likelihood_func(self, X, y, label):
        sample_mean = X[y == label].mean()
        sample_std = X[y == label].std()
        sample_std = np.where(sample_std == 0, 1e-9, sample_std)

        def output(X):
            """
            actually, the proba is not probability; its pdf so can be > 1. but we just want to compare
            between classes to get the prediction and do not need the actual probabilities
            (for the same reason as not computing the normalizing denominator) so just using pdf is okay.
            
            Laplace smoothing is not applied here since we are not dealing with counts or discrete values
            hence we do not need to worry about zero probabilities.
            """
            normal_pdf_values = scipy.stats.norm.pdf(X, loc=sample_mean, scale=sample_std)
            probas = pd.DataFrame(normal_pdf_values, index=X.index, columns=X.columns)
            return pd.Series(probas.prod(axis=1), index=X.index)
        
        return output

    def _check_input(self, X):
        if not all([pd.api.types.is_numeric_dtype(X[col]) for col in X.columns]): # can only check for each column
            raise TypeError("Input must be a numeric matrix.")
