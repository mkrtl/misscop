from copula_modelling.marginal_models.base import MarginalModel
from learning_methods.GMM import GMM
from scipy.optimize import minimize, LinearConstraint
from scipy import stats
import numpy as np


class MarginalMixtureModel(MarginalModel):
    """
    Super class for marginal mixture models.
    They have to have all the methods that are defined below:
    """
    def __init__(self, params, bandwidth, verbose=False):
        super().__init__(params, verbose=verbose)
        self.bandwidth = bandwidth
        self.n_mixture_marginal = params.shape[0]
        self._num_samples_cdf_inverse = 10000
        self._sample_for_cdf_inverse = self.sample(self._num_samples_cdf_inverse)
        self.name = None

    def single_dist(self, mu):
        raise NotImplementedError

    def single_dist_pdf_derivative_mu(self, x, mu):
        raise NotImplementedError

    def single_dist_cdf_derivative_mu(self, x, mu):
        raise NotImplementedError

    def loglikelihood(self, data, mus, verbose=None):
        if verbose is None:
            verbose = self.verbose
        col = data[~np.isnan(data)]
        pdf_values = self.pdf(col, mus=mus)
        if verbose:
            print("Current value for mus  {}".format(mus))
            print("Likelihood of column: {}".format(np.sum(np.log(pdf_values))))
        return np.sum(np.log(pdf_values))

    def derivative_loglikelihood(self, data, mus, verbose=None):
        if verbose is None:
            verbose = self.verbose
        col = data[~np.isnan(data)]
        if verbose:
            print("Derivative of Likelihood of column: {}".format(np.sum(np.array(1 / self.pdf(col, mus=mus))
                                                                         * self.pdf_derivative_mu(col, mus=mus),
                                                                         axis=1)
                                                                  )
                  )
        return np.sum(np.array(1 / self.pdf(col, mus=mus)) * self.pdf_derivative_mu(col, mus=mus), axis=1)

    def maximize_likelihood(self, data, verbose=None):
        if verbose is None:
            verbose = self.verbose

        # Remove null values from column
        col = data[~np.isnan(data)]

        constraint_matrix, lower_bound, upper_bound = self.constraint
        wrapper_func_to_optimize = lambda mus_lbda: -self.loglikelihood(col, mus_lbda, verbose=verbose)
        wrapper_grad = lambda mus_lbda: -self.derivative_loglikelihood(col, mus_lbda, verbose=verbose)

        lower_bound = - np.inf * np.ones(self.n_mixture_marginal)
        upper_bound = np.zeros(self.n_mixture_marginal)
        final_constraint = LinearConstraint(constraint_matrix, lower_bound, upper_bound)
        mu_start_finder = GMM(n_components=self.n_mixture_marginal,
                              params="m",
                              init_params="m",
                              covariance_type="spherical")
        mu_start_finder.fit(col[~np.isnan(col)].reshape(-1, 1))
        mu_start = mu_start_finder.means_.reshape(self.n_mixture_marginal)
        mu_start = np.sort(mu_start)
        if verbose:
            print("Log-Llikelihood value after initializing mus {}".format(self.loglikelihood(col, mu_start)))
            print("Mus are {}".format(mu_start))
        minimizer_results = minimize(wrapper_func_to_optimize, mu_start,
                                     jac=wrapper_grad,
                                     constraints=final_constraint,
                                     )
        return minimizer_results.x

    @property
    def constraint(self):
        """
        This gives constraint, upper bound and lower bound that guarantees that the mus are ordered.
        """
        constraint_matrix = np.zeros((self.n_mixture_marginal, self.n_mixture_marginal))
        constraint_matrix += np.eye(self.n_mixture_marginal)
        constraint_matrix -= np.eye(self.n_mixture_marginal, k=1)
        # No constraint on the last theta_jg and on the first theta_j1
        constraint_matrix[-1, -1] = 0
        lower_bound = -np.inf * np.ones(self.n_mixture_marginal)
        upper_bound = np.zeros(self.n_mixture_marginal)
        return constraint_matrix, lower_bound, upper_bound

    def cdf_inverse(self, probs):
        probs_is_nan = np.isnan(probs)
        sample_values = self._sample_for_cdf_inverse
        existing_probs = probs[~probs_is_nan]
        result = np.zeros_like(probs)
        # Percentile estimator as described here:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
        result[~probs_is_nan] = stats.mstats.mquantiles(sample_values,
                                                        existing_probs,
                                                        alphap=0,
                                                        betap=0)
        result[probs_is_nan] = np.nan
        return result

    def set_params(self, params):
        self.params = params
        self._sample_for_cdf_inverse = self.sample(self._num_samples_cdf_inverse)

    @staticmethod
    def _make_index(k, size):
        # Forked from
        # https://github.com/statsmodels/statsmodels/blob/2228133e6d4b9724730b046e8759a0e9d133c366/statsmodels/distributions/mixture_rvs.py#L3
        """
        Returns a boolean index for every row where every column is chosen with probability (1/k).
        """
        rv = np.random.uniform(size=(size, 1))
        cumprob = np.cumsum(k * [1 / k])
        return np.logical_and(np.r_[0, cumprob[:-1]] <= rv, rv < cumprob)

    def sample(self, n):
        """Forked from
        https://github.com/statsmodels/ \
        statsmodels/blob/2228133e6d4b9724730b046e8759a0e9d133c366/statsmodels/distributions/mixture_rvs.py#L3
        """

        idx = self._make_index(self.n_mixture_marginal, n)
        sample = np.zeros(n)
        for i in range(self.n_mixture_marginal):
            sample_idx = idx[..., i]
            sample_size = sample_idx.sum()
            loc = self.params[i]
            dist = self.single_dist(loc)
            sample[sample_idx] = dist.rvs(size=sample_size)
        return sample


class GaussianMixture(MarginalMixtureModel):

    def __init__(self, params, bandwidth, verbose=False):
        super().__init__(params, bandwidth, verbose=verbose)
        self.name = "gaussian_mixture"

    def single_dist(self, mu):
        return stats.norm(loc=mu, scale=self.bandwidth)

    def single_dist_pdf_derivative_mu(self, x, mu):
        return self.single_dist(mu).pdf(x) * (x - mu) / self.bandwidth ** 2

    def cdf(self, x):
        values_cdfs = (1 / self.n_mixture_marginal) * np.array([self.single_dist(mu).cdf(x) for mu in self.params])
        return np.sum(values_cdfs, axis=0)

    def single_dist_cdf_derivative_mu(self, x, mu):
        return -self.single_dist(mu).pdf(x)

    def pdf(self, x, mus=None):
        if mus is None:
            mus = self.params
        return (1 / self.n_mixture_marginal) * np.sum(np.array([self.single_dist(mu).pdf(x) for mu in mus]),
                                                      axis=0)

    def pdf_derivative_mu(self, x, mus=None):
        if mus is None:
            mus = self.params
        return (1 / self.n_mixture_marginal) * np.array([self.single_dist_pdf_derivative_mu(x, mu)
                                                         for mu in mus])

    def cdf_derivative_mu(self, x, mus=None):
        if mus is None:
            mus = self.params
        return (1 / self.n_mixture_marginal) * np.array([self.single_dist_cdf_derivative_mu(x, mu) for mu in mus])


class EpanechnikovMixture(MarginalMixtureModel):

    def __init__(self, params, bandwidth):
        super().__init__(params, bandwidth)
        self.name = "epanechnikov_mixture"

    def single_dist(self, mu):
        """
        :param mu:
        :return:
        """
        """
        c = 4 corresponds to epanechikov distribution:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rdist.html
        """
        return stats.rdist(4, loc=mu, scale=self.bandwidth)

    def single_dist_pdf_derivative_mu(self, x, mu):
        return np.where(np.abs(x - mu) < self.bandwidth, - (8 / 3) * ((mu - x) / self.bandwidth ** 2),
                        0)

    def cdf(self, x):
        values_cdfs = (1 / self.n_mixture_marginal) * np.array([self.single_dist(mu).cdf(x) for mu in self.params])
        return np.sum(values_cdfs, axis=0)

    def single_dist_cdf_derivative_mu(self, x, mu):
        return -self.single_dist(mu).pdf(x)

    def pdf(self, x):
        return np.sum((1 / self.n_mixture_marginal) * np.array([self.single_dist(mu).pdf(x) for mu in self.params]),
                      axis=0)

    def pdf_derivative_mu(self, x):
        return (1 / self.n_mixture_marginal) * np.array([self.single_dist_pdf_derivative_mu(x, mu)
                                                         for mu in self.params])

    def cdf_derivative_mu(self, x):
        return (1 / self.n_mixture_marginal) * np.array([-self.single_dist_cdf_derivative_mu(x, mu)
                                                         for mu in self.params])

    def cdf_inverse(self, probs):
        probs_is_nan = np.isnan(probs)
        sample_values = self._sample_for_cdf_inverse
        existing_probs = probs[~probs_is_nan]
        result = np.zeros_like(probs)
        result[~probs_is_nan] = stats.mstats.mquantiles(sample_values, existing_probs)
        result[probs_is_nan] = np.nan
        return result


class GaussianMixtureAdapted(MarginalMixtureModel):

    def __init__(self, params, bandwidth, weights):
        self.weights = weights
        super().__init__(params, bandwidth)
        self.thetas = list(zip(self.params, self.bandwidth))
        self.name = "gaussianadapted"

    def single_dist(self, mu, bandwidth):
        return stats.norm(loc=mu, scale=bandwidth)

    def single_dist_pdf_derivative_mu(self, x, mu, bandwidth):
        return self.single_dist(mu, bandwidth).pdf(x) * (x - mu) / bandwidth ** 2

    def cdf(self, x):
        values_cdfs = self.weights @ np.array([self.single_dist(mu, bandwidth).cdf(x)
                                               for mu, bandwidth in self.thetas])
        return values_cdfs

    def single_dist_cdf_derivative_mu(self, x, mu, bandwidth):
        return -self.single_dist(mu, bandwidth).pdf(x)

    def pdf(self, x):
        return self.weights @ np.array([self.single_dist(mu, bandwidth).pdf(x) for mu, bandwidth in self.thetas])

    def pdf_derivative_mu(self, x):
        return np.repeat(self.weights.reshape(-1, 1), x.shape[0], axis=1) * np.array([self.single_dist_pdf_derivative_mu(x, mu, bandwidth)
                                                         for mu, bandwidth in self.thetas])

    def cdf_derivative_mu(self, x):
        return np.repeat(self.weights.reshape(-1, 1), x.shape[0], axis=1) * np.array([self.single_dist_cdf_derivative_mu(x, mu, bandwidth)
                                                         for mu, bandwidth in self.thetas])

    def sample(self, n):
        """Forked from
        https://github.com/statsmodels/ \
        statsmodels/blob/2228133e6d4b9724730b046e8759a0e9d133c366/statsmodels/distributions/mixture_rvs.py#L3
        """

        idx = self._make_index(self.weights, n)
        sample = np.zeros(n)
        for i in range(self.n_mixture_marginal):
            sample_idx = idx[..., i]
            sample_size = sample_idx.sum()
            loc = self.params[i]
            bandwith = self.bandwidth[i]
            dist = self.single_dist(loc, bandwith)
            sample[sample_idx] = dist.rvs(size=sample_size)
        return sample

    @staticmethod
    def _make_index(weights, size):
        # Forked from
        # https://github.com/statsmodels/statsmodels/blob/2228133e6d4b9724730b046e8759a0e9d133c366/statsmodels/distributions/mixture_rvs.py#L3
        """
        Returns a boolean index for every row where every column is chosen with probability (1/k).
        """
        rv = np.random.uniform(size=(size, 1))
        cumprob = np.cumsum(weights)
        return np.logical_and(np.r_[0, cumprob[:-1]] <= rv, rv < cumprob)

    def cdf_inverse(self, probs):
        probs_is_nan = np.isnan(probs)
        sample_values = self._sample_for_cdf_inverse
        existing_probs = probs[~probs_is_nan]
        result = np.zeros_like(probs)
        # Percentile estimator as described here:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
        result[~probs_is_nan] = stats.mstats.mquantiles(sample_values,
                                                        existing_probs,
                                                        alphap=0,
                                                        betap=0)
        result[probs_is_nan] = np.nan
        return result
