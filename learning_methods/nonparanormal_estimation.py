from scipy.stats import norm
from scipy import stats
import numpy as np


class EmpiricalMarginalCDFEstimator:

    def __init__(self, np_array, method="truncation", delta_n=None):

        self.existing_methods = ["truncation", "shrinkage"]
        self.array = np_array
        self.idx_not_none = ~np.isnan(np_array)
        self.non_null_values = np_array[self.idx_not_none]
        self.num_non_null_values = len(self.non_null_values)
        self.order = np.argsort(self.non_null_values)
        if method in self.existing_methods:
            if method == "truncation":
                self.method = "truncation"
            elif method == "shrinkage":
                self.method = "shrinkage"
        else:
            raise NotImplementedError("Method is not implemented. Use one of {}"
                                      .format(self.existing_methods))
        if delta_n is None:
            self.delta_n = 1 / (4 * self.num_non_null_values ** (1 / 4) *
                                np.sqrt(np.pi * np.log(self.num_non_null_values)))
        else:
            self.delta_n = delta_n
        self.ranks = None

    def calculate_ecd_marginal(self):
        if self.ranks is None:
            order = self.order
            ranks = np.empty_like(self.non_null_values)
            if self.method == "truncation":
                ranks[order] = (np.arange(1, self.num_non_null_values + 1) / self.num_non_null_values)
                self.ranks = np.minimum(np.maximum(ranks, self.delta_n), 1 - self.delta_n)
            elif self.method == "shrinkage":
                ranks[order] = (np.arange(1, self.num_non_null_values + 1) / (self.num_non_null_values + 1))
                self.ranks = ranks
        return self.ranks

    def gaussianize_data(self):
        normalized_vec = np.zeros_like(self.array)
        if self.ranks is None:
            ranks = self.calculate_ecd_marginal()
        else:
            ranks = self.ranks
        normalized_vec[self.idx_not_none] = norm.ppf(ranks)
        normalized_vec[~self.idx_not_none] = np.nan
        # Choose ddof=1 as in the huge package of R
        # Normalization is also done as in the R package.
        return normalized_vec / np.nanstd(normalized_vec, ddof=1)

    def empirical_percentile_function(self, q, method="linear"):
        """
        Calculate the empirical percentile function using different methods.
        :param method: Method for estimating the empirical percentile function
        :type method: str
        :return: Function
        """
        return_array = np.zeros_like(q)
        if self.ranks is None:
            self.calculate_ecd_marginal()
        if method == "linear":
            max_rank = np.max(self.ranks)
            min_rank = np.min(self.ranks)
            return_array[q >= max_rank] = np.max(self.non_null_values)
            return_array[q <= min_rank] = np.min(self.non_null_values)
            return_array[(q <= max_rank) & (q >= min_rank)] = np.array(list(map(self._weigh_closest_points_linear,
                                                                                q[(q <= max_rank) & (q >= min_rank)])))
            return return_array

    def _weigh_closest_points_linear(self, q_element):
        if self.ranks is None:
            self.calculate_ecd_marginal()
        # Get two closest observations:
        rank_bigger = self.ranks >= q_element
        rank_smaller = self.ranks <= q_element
        closest_bigger_rank = np.min(self.ranks[rank_bigger])
        closest_smaller_rank = np.max(self.ranks[rank_smaller])

        bigger_value = self.non_null_values[self.ranks == closest_bigger_rank][0]
        smaller_value = self.non_null_values[self.ranks == closest_smaller_rank][0]

        # Now get the weighted mean of the two closest values:
        return ((q_element - closest_smaller_rank) / (closest_bigger_rank - closest_smaller_rank)) \
               * (bigger_value - smaller_value) \
               + smaller_value


def sample_copula(data, sigma, n, marginal_cdf_method="truncation", percentile_method="linear"):
    """
    :param percentile_method:
    :param data:
    :param sigma:
    :param n:
    :param marginal_cdf_method:
    :return:
    """
    # Calculate the ecdf:
    marginals = []
    for col_number in range(data.shape[1]):
        col = data[~np.isnan(data[:, col_number]), col_number]
        curr_marginal = EmpiricalMarginalCDFEstimator(col, method=marginal_cdf_method)
        marginals.append(curr_marginal)

    gaussian_copula_sample = stats.norm.cdf(stats.multivariate_normal(mean=None, cov=sigma).rvs(n))
    final_sample = np.zeros_like(gaussian_copula_sample)
    for col_number in range(gaussian_copula_sample.shape[1]):
        col_gaussian_sample = gaussian_copula_sample[:, col_number]
        # Now get the maximal value in the ecdf that is smaller to the sample:
        curr_marginal = marginals[col_number]
        final_sample[:, col_number] = curr_marginal.empirical_percentile_function(col_gaussian_sample,
                                                                                  method=percentile_method)
    return final_sample
