from functools import partial
import time
from scipy.linalg import pinvh
from learning_methods.EMGlasso import EMGlasso, EMCorrelation
from copula_modelling.marginal_models import mixture
from learning_methods.constrained_covariance import ConstrainedCovarianceEstimator
import numpy as np
from scipy import stats
from scipy.optimize import minimize, LinearConstraint


class GaussianCopula(object):
    def __init__(self, data,
                 initial_sigma, M, initial_params=None,
                 marginal_distributions=None,
                 lambda_glasso=0,
                 eps=0.01,
                 max_iter=25,
                 verbose=False,
                 track_thetas=False,
                 increasing_M=False,
                 weights_marginal=None,
                 sigma_inverse_constraint=None,
                 sigma_finder="approximate"):
        self.data = data
        self.num_dims = self.data.shape[1]
        self.num_rows = self.data.shape[0]
        self.marginal_distributions = marginal_distributions
        self.sigma = initial_sigma
        self.n_parameters_marginal_total = np.sum([marginal.n_params for marginal in self.marginal_distributions])
        if initial_params is None:
            self.params = [None for _ in range(self.num_dims)]
        else:
            self.params = initial_params
        if len(self.marginal_distributions) != self.num_dims:
            raise Exception("There has to be the same number of distributions as number of columns!")
        self.lambda_glasso = lambda_glasso
        self.verbose = verbose
        if isinstance(M, np.ndarray):
            self.M_array = M
            self.M = self.M_array[0]
        else:
            self.M = M
            self.M_array = None
        self.eps = eps
        for k, dist in enumerate(self.marginal_distributions):
            is_mixture = False
            if isinstance(dist, mixture.MarginalModel):
                is_mixture = True
            if is_mixture is True and self.params[k] is None:
                self.params[k] = dist.maximize_likelihood(data, verbose=self.verbose)
            else:
                self.params[k] = dist.params
        self.track_thetas = track_thetas
        if self.track_thetas:
            self.tracked_thetas = []
        self.increasing_M = increasing_M
        if self.increasing_M:
            self.increasing_factor = (4 / 3)
        self.max_iter = max_iter
        self.weights_marginal = weights_marginal
        self.sigma_finder = sigma_finder
        if sigma_inverse_constraint is not None:
            self.inverse_constraint_exists = True
        else:
            self.inverse_constraint_exists = False
        self.sigma_inverse_constraint = sigma_inverse_constraint

    @staticmethod
    def conditional_z(z_missing_and_observed, cov):
        # TODO: Apply this method on Z and use Missing Patterns to calculate inverses only once.
        """
        Calculates the copula for z_missing | z_observed. We can then use measurability arguments such that we can use
        the conditional distribution of z_missing | z_observed for sampling x_missing | x_observed.
        Returns mean and the covariance at Phi ^(-1) (u).
        """
        z_is_nan = np.isnan(z_missing_and_observed)
        z_existing = z_missing_and_observed[~z_is_nan]
        sigma_existing_existing = cov[np.ix_(~z_is_nan, ~z_is_nan)]
        sigma_missing_missing = cov[np.ix_(z_is_nan, z_is_nan)]
        sigma_missing_existing = cov[np.ix_(z_is_nan, ~z_is_nan)]
        inv_sigma_existing_existing = pinvh(sigma_existing_existing)
        expectation_hidden_given_visible = sigma_missing_existing @ inv_sigma_existing_existing @ z_existing
        covariance_hidden_given_visible = sigma_missing_missing \
                                          - sigma_missing_existing \
                                          @ inv_sigma_existing_existing \
                                          @ sigma_missing_existing.T

        return expectation_hidden_given_visible, covariance_hidden_given_visible

    def maximize_sigma(self, X):
        Z = self.transform_X_to_Z(X)
        sigma = self.sigma.copy()
        mu = np.zeros(self.num_dims)
        if self.inverse_constraint_exists or self.sigma_finder == "approximate":
            sigma_optimizer = EMGlasso(Z,
                                       sigma_init=sigma,
                                       lambda_glasso=self.lambda_glasso,
                                       max_iter=1000,
                                       eps=10 ** (-5))
            sufficient_statistics = sigma_optimizer.e_step(mu, sigma)

            if self.inverse_constraint_exists:
                mu_estimate = sufficient_statistics[0] / self.num_rows
                sigma_estimate = (sufficient_statistics[1] / self.num_rows) - np.outer(mu_estimate, mu_estimate)
                estimator = ConstrainedCovarianceEstimator(sigma_estimate.copy(),
                                                           self.sigma_inverse_constraint,
                                                           eps=10 ** (-10))
                estimated_sigma, _ = estimator.run_algorithm()
            else:
                _, estimated_sigma = sigma_optimizer.m_step(sufficient_statistics[0],
                                                            sufficient_statistics[1])

            # Rescale covariance matrix to correlation matrix:
            rescaling_matrix = 1 / np.sqrt(np.outer(np.diagonal(estimated_sigma), np.diagonal(estimated_sigma)))
            return np.multiply(estimated_sigma, rescaling_matrix)
        elif self.sigma_finder == "exact":
            if self.lambda_glasso != 0:
                raise Exception("You should not apply the Graphical Lasso for the exact correlation matrix")
            corr_finder = EMCorrelation(Z, sigma)
            sufficient_statistics = corr_finder.e_step(mu, sigma)
            _, estimated_sigma = corr_finder.m_step(sufficient_statistics[0],
                                                    sufficient_statistics[1])
            return estimated_sigma
        else:
            raise ValueError("You have to provide either 'exact' or 'approximate' as value for argument sigma_finder.")

    def augment_z_data(self, Z, cov):
        augmented_data = np.zeros((self.num_rows * self.M, self.num_dims))
        # TODO: Find missing patterns and then do the augmentation vectorized!
        for k, row in enumerate(Z):
            row_is_nan = np.isnan(row)
            if np.sum(row_is_nan) > 0:
                conditional_mean, conditional_cov = self.conditional_z(row, cov)
                sample_z_missing = stats.multivariate_normal.rvs(conditional_mean, conditional_cov, size=self.M)
                augmented_data[k * self.M:(k + 1) * self.M, row_is_nan] = sample_z_missing.reshape(
                    (augmented_data[k * self.M:(k + 1) * self.M, row_is_nan]).shape)
                augmented_data[k * self.M:(k + 1) * self.M, ~row_is_nan] = row[~np.isnan(row)]
            else:
                augmented_data[k * self.M:(k + 1) * self.M, :] = row
        return augmented_data

    @staticmethod
    def _get_missing_patterns(data):
        # This method shall be used later in order to calculate the conditional distribution
        # mapper only once.
        data_is_nan = np.isnan(data)
        # Get all missing patterns:
        missing_data_patterns = np.unique(data_is_nan, axis=0)
        return missing_data_patterns

    def transform_X_to_Z(self, X, params=None):
        """
        num_dims = self.num_dims
        Z = np.zeros_like(X)
        for column in range(num_dims):
            curr_marginal = self.marginal_distributions[column]
            if params is not None:
                curr_marginal = self.set_params_of_marginal(column, params[column])
            Z[:, column] = stats.norm.ppf(curr_marginal.cdf(X[:, column]))
        """
        U = self.transform_X_to_U(X, params=params)
        return stats.norm.ppf(U)

    def transform_X_to_U(self, X, params=None):
        num_dims = self.num_dims
        U = np.zeros_like(X)
        for column in range(num_dims):
            curr_marginal = self.marginal_distributions[column]
            if params is not None:
                curr_marginal = self.set_params_of_marginal(column, params[column])
            U[:, column] = curr_marginal.cdf(X[:, column])
        return U

    def transform_Z_to_X(self, Z, params=None):
        X = np.zeros_like(Z)
        for column in range(self.num_dims):
            if params is not None:
                curr_marginal = self.set_params_of_marginal(column, params[column])
            else:
                curr_marginal = self.marginal_distributions[column]
            X[:, column] = curr_marginal.cdf_inverse(stats.norm.cdf(Z[:, column]))
        return X

    def set_params_of_marginal(self, col_number, params):
        marginal = self.marginal_distributions[col_number]
        marginal.set_params(params)
        return marginal

    def minus_q_theta(self, augmented_X, params, precision=None):
        params = params.copy()
        if precision is None:
            precision = pinvh(self.sigma)
        num_cols = augmented_X.shape[1]

        Z = self.transform_X_to_Z(augmented_X, params=params)
        first_func = - (1 / (2 * self.M)) * np.sum(np.tensordot(Z, ((precision - np.eye(num_cols)) @ Z.T).T))
        marginals = [self.set_params_of_marginal(k, params[k]) for k in range(num_cols)]
        second_func = (1 / self.M) * np.sum(np.log([curr_marginal.pdf(augmented_X[:, i]) for i, curr_marginal
                                                    in enumerate(marginals)]))
        if self.verbose:
            print("Function evaluation")
            print(- (first_func + second_func))
        return - (first_func + second_func)

    def grad_minus_q_theta(self, augmented_X, params, precision=None):
        if precision is None:
            precision = pinvh(self.sigma)

        num_cols = augmented_X.shape[1]
        num_rows = augmented_X.shape[0]

        marginals = [self.set_params_of_marginal(k, params[k]) for k in range(num_cols)]
        Z = self.transform_X_to_Z(augmented_X, params=params)
        # Dh = np.zeros((num_rows, num_cols, num_cols * self.n_mixture_marginal))
        Dh = np.zeros((num_rows, num_cols, self.n_parameters_marginal_total))
        diag_elements_h = []
        for k in range(num_cols):
            diag_elements_h.append(marginals[k].cdf_derivative_mu(augmented_X[:, k]) / \
                                   stats.norm.pdf(stats.norm.ppf(marginals[k].cdf(augmented_X[:, k]))))
        param_index = 0
        for k in range(num_cols):
            n_params_for_marginal_k = marginals[k].n_params
            # Workaround if the diag_elements of k are only one-dimensional:
            if diag_elements_h[k].ndim == 1:
                Dh[:, k, param_index:(param_index + n_params_for_marginal_k)] \
                    = (diag_elements_h[k]).T.reshape(-1, 1)
            else:
                Dh[:, k, param_index:(param_index + n_params_for_marginal_k)] \
                    = (diag_elements_h[k]).T
            param_index += n_params_for_marginal_k

        gradient_first_func = np.tensordot(Z @ (np.eye(num_cols) - precision), Dh)
        gradient_second_func = np.zeros_like(gradient_first_func)
        param_index = 0
        for k in range(num_cols):
            n_params_for_marginal_k = marginals[k].n_params
            axis = 1 if n_params_for_marginal_k > 1 else 0
            gradient_second_func[param_index:(param_index + n_params_for_marginal_k)] = \
                np.sum(1 / marginals[k].pdf(augmented_X[:, k]) * marginals[k].pdf_derivative_mu(augmented_X[:, k]),
                       # Workaround if the diag_elements of k are only one-dimensional:
                       axis=axis)
            param_index += n_params_for_marginal_k

        if self.verbose:
            print("Derivative evaluation")
            print(np.linalg.norm(-(gradient_first_func + gradient_second_func) / self.M))
        return -(gradient_first_func + gradient_second_func) / self.M

    @property
    def _ordered_theta_constraint(self):
        constraint_matrix = np.zeros((self.n_parameters_marginal_total, self.n_parameters_marginal_total))
        lower_bound = np.zeros(self.n_parameters_marginal_total)
        upper_bound = np.zeros(self.n_parameters_marginal_total)
        idx = 0
        for dist in self.marginal_distributions:
            curr_constraint, curr_lower_bound, curr_upper_bound = dist.constraint
            constraint_matrix[idx:idx + dist.n_params, idx:idx + dist.n_params] = curr_constraint
            lower_bound[idx:idx + dist.n_params] = curr_lower_bound
            upper_bound[idx:idx + dist.n_params] = curr_upper_bound
            idx = idx + dist.n_params
        # If there are no constraints, then apply no constraint.
        if np.max(constraint_matrix) == np.min(constraint_matrix):
            return None
        else:
            return LinearConstraint(constraint_matrix, lower_bound, upper_bound)

    def _reshape_params(self, params):
        return_param_list = []
        idx = 0
        if np.ndim(params) == 1:
            for k in range(len(self.marginal_distributions)):
                curr_marginal_dist = self.marginal_distributions[k]
                return_param_list.append(params[idx:idx + curr_marginal_dist.n_params])
                idx = idx + curr_marginal_dist.n_params
            return return_param_list
        else:
            return params

    def maximize_q_theta(self, augmented_X, params, precision=None):
        TOL_MINIMIZER = 0.001
        if precision is None:
            precision = pinvh(self.sigma)

        func_to_optimize = partial(self.minus_q_theta, augmented_X,
                                   precision=precision)

        wrapper_func_to_optimize = lambda params: func_to_optimize(self._reshape_params(params))

        grad = partial(self.grad_minus_q_theta, augmented_X, precision=precision)
        wrapper_grad = lambda params: grad(self._reshape_params(params))
        constraint = self._ordered_theta_constraint
        minimizer_results = minimize(wrapper_func_to_optimize, np.hstack(params),
                                     jac=wrapper_grad,
                                     tol=TOL_MINIMIZER,
                                     constraints=constraint,
                                     options={"disp": self.verbose})
        return minimizer_results.x

    def run_em_algo(self):
        num_runs = 1
        eps = np.inf
        while eps > self.eps and num_runs <= self.max_iter:
            sigma_old = np.copy(self.sigma)
            # Maximize sigma:
            new_sigma = np.copy(self.maximize_sigma(self.data))
            precision = pinvh(new_sigma)
            # E-Step:
            Z = self.transform_X_to_Z(self.data)
            # Sample with respect to old sigma!
            augmented_Z = self.augment_z_data(Z, self.sigma)
            augmented_X = self.transform_Z_to_X(augmented_Z)
            # M-Step
            # Maximize q_theta
            params_old = self.params.copy()
            start = time.time()
            # Now maximize the samples resulting from old sigma while using new sigma!
            self.params = self.maximize_q_theta(augmented_X, params_old, precision=precision)
            if np.ndim(self.params) == 1:
                self.params = self._reshape_params(self.params)
            for k in range(self.num_dims):
                self.marginal_distributions[k].set_params(self.params[k])
            if self.track_thetas:
                self.tracked_thetas.append(self.params)
            end = time.time()
            if self.verbose:
                print("Current loop took {} seconds".format(end - start))
            self.sigma = new_sigma
            if num_runs > 1:
                diff_sigmas = np.linalg.norm(self.sigma - sigma_old) / np.linalg.norm(sigma_old)
                # Take the mean of the differences of the parameters of all dimensions
                diff_thetas = np.mean([np.linalg.norm(self.params[k] - params_old[k]) for k in range(self.num_dims)])
                eps = diff_thetas + diff_sigmas
                if self.verbose:
                    print("Difference between the sigmas: {}".format(diff_sigmas))
                    print("Difference between the thetas: {}".format(diff_thetas))
                    print("Current sigma: {}".format(self.sigma))
            num_runs += 1
            if isinstance(self.M_array, np.ndarray):
                if len(self.M_array) < num_runs:
                    break
                else:
                    self.M = self.M_array[num_runs - 1]
            elif self.increasing_M:
                self.M = self.M + int((4 / 3) * num_runs)

        return self.sigma, self.params

    def sample(self, n):
        """

        :param n: number of samples
        :type n: int
        :return:
        """
        if self.sigma is None:
            raise UserWarning("You have to run the EM algorithm first!")

        else:
            sigma = self.sigma
            normal_sample = stats.multivariate_normal(mean=None, cov=sigma).rvs(n)
            return self.transform_Z_to_X(normal_sample)

    def density(self, x):
        """

        :param x: N x p matrix where every row is a point where we evaluate the density.
        :return: density values
        """
        if self.sigma is None:
            raise UserWarning("You have to run the EM algorithm first!")

        U = self.transform_X_to_U(x)
        density_estimates = np.zeros(x.shape)

        for k in range(self.num_dims):
            density_estimates[:, k] = self.marginal_distributions[k].pdf(x[:, k])

        copula_density_values = self.gaussian_copula_density(U)
        return copula_density_values * np.prod(density_estimates, axis=1)

    def gaussian_copula_density(self, u):
        """

        :param u: N x p matrix where every row is a point where we evaluate the density.
        :return:
        """
        if self.sigma is None:
            raise UserWarning("You have to run the EM algorithm first!")

        sigma = self.sigma

        det_sigma = np.linalg.det(sigma)
        precision = np.linalg.inv(sigma)
        z = stats.norm.ppf(u)

        return (1/det_sigma) * np.exp(-(1/2) * np.sum((z @ (precision - np.eye(self.num_dims))) * z, axis=1))




