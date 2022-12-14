import numpy as np
from matplotlib import pyplot as plt


class MarginalModel(object):
    """
    Super class for marginal distributions.
    It has to have all the methods that are defined below:
    """
    def __init__(self, params, verbose=False):
        if isinstance(params, (int, float)):
            self.n_params = 1
        else:
            self.n_params = params.shape[0]
        self.verbose = verbose
        self.params = params

    def cdf(self, x):
        raise NotImplementedError

    def cdf_inverse(self, probs):
        raise NotImplementedError

    def pdf(self, x):
        raise NotImplementedError

    def pdf_derivative_mu(self, x):
        raise NotImplementedError

    def cdf_derivative_mu(self, x):
        raise NotImplementedError

    def plot_cdf(self, grid=np.linspace(-3, 3, 1000), **kwargs):
        plt.plot(grid, self.cdf(grid), **kwargs)

    def plot_pdf(self, grid=np.linspace(-3, 3, 1000), **kwargs):
        plt.plot(grid, self.pdf(grid), **kwargs)

    def sample(self, n):
        raise NotImplementedError

    def set_params(self, params):
        raise NotImplementedError

    @property
    def constraint(self):
        """
        This gives constraint, upper bound and lower bound that guarantees that the mus are ordered.
        In general there are no constraints on the parameters. Applies only for mixture models.
        """
        constraint_matrix = np.zeros((self.n_params, self.n_params))
        lower_bound = -np.inf * np.ones(self.n_params)
        upper_bound = np.inf * np.ones(self.n_params)
        return constraint_matrix, lower_bound, upper_bound
