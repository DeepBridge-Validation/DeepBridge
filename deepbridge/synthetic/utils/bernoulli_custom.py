import numpy as np
from copulas.univariate.base import Univariate


class BernoulliCustom(Univariate):
    def __init__(self):
        self.p = None

    def fit(self, X):
        """
        Fit Bernoulli distribution to data X.

        Args:
            X (array-like): Data with binary values (0 or 1).
        """
        X = np.asarray(X)
        if not np.all(np.isin(X, [0, 1])):
            raise ValueError(
                'Bernoulli distribution only accepts binary data (0 and 1).'
            )
        self.p = np.mean(X)

    def probability_density(self, X):
        """
        Compute the probability mass function for given values.

        Args:
            X (array-like): Data points to evaluate PMF.

        Returns:
            array-like: Probability values.
        """
        X = np.asarray(X)
        return np.where(X == 1, self.p, 1 - self.p)

    def cumulative_distribution(self, X):
        """
        Compute cumulative distribution function for given values.

        Args:
            X (array-like): Data points to evaluate CDF.

        Returns:
            array-like: Cumulative probability values.
        """
        X = np.asarray(X)
        return np.where(X < 1, 1 - self.p, 1.0)

    def sample(self, n_samples=1):
        """
        Generate random samples from the Bernoulli distribution.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            np.array: Array of binary samples.
        """
        return np.random.binomial(1, self.p, size=n_samples)

    def to_dict(self):
        """
        Serialize distribution parameters to a dictionary.

        Returns:
            dict: Dictionary containing distribution parameters.
        """
        return {'type': 'custom.BernoulliCustom', 'p': self.p}

    @classmethod
    def from_dict(cls, params):
        """
        Create BernoulliCustom instance from serialized parameters.

        Args:
            params (dict): Parameters for initializing BernoulliCustom.

        Returns:
            BernoulliCustom: Instance with parameters set.
        """
        instance = cls()
        instance.p = params['p']
        return instance
