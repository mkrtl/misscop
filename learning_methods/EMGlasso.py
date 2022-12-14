import numpy as np
from sklearn.covariance import graphical_lasso, shrunk_covariance
from sklearn.model_selection import KFold

from learning_methods.optimal_correlation import OptimalCorrelationFinder


class EMGlasso(object):
    def __init__(self, data, sigma_init,
                 lambda_glasso=0.0,
                 eps=0.01,
                 calculate_log_likelihood=False,
                 solver="cd",
                 max_iter=20,
                 eps_glasso=np.finfo(np.float64).eps,
                 ):
        """

        :param data: data with missing entries
        :type data: numpy.array
        :param sigma_init:
        :param lambda_glasso:
        :param eps:
        """
        self.data = data
        self.sigma_init = sigma_init
        self.n_obs = data.shape[0]
        self.n_dims = data.shape[1]
        self.indexes_missing = np.argwhere(np.isnan(self.data))
        self.eps = eps
        self.lambda_glasso = lambda_glasso
        self.estimated_sparse_cov = None
        self.missing_values = np.ma.masked_invalid(self.data).mask
        self.log_likelihood_result = None
        self.calculate_log_likelihood = calculate_log_likelihood
        self.log_likelihoods = []
        self.solver = solver
        self.max_iter = max_iter
        self.eps_glasso = eps_glasso

    def e_step(self, mu, sigma_t):
        """
            This is the E-Step of the EM-algorithm: we calculate the expected covariance under the current estimation
            of theta_t (in this case it is only sigma, as we assume mu to be 0).
            Note that we just calculate the expected sufficient statistic given the observed data! We are not
            calculating the values of the missing values explicitly! We do not need them for the M-step, that is why we
            just skip it.
            See https://arxiv.org/pdf/0903.5463.pdf for further details.
            :param mu:
            :param sigma_t: current estimation of the sigma
            :type sigma_t: numpy.array
            :return: sigma containing the expected sigma given the current sigma_t
            """
        # Iterate through all observations
        return self.calculate_sufficient_statistics(mu, sigma_t)

    def m_step(self, mu_t, sigma_t):
        """
        :param sigma_t:
        :return:
        """
        # For the maximum likelihood estimation we avoid the graphical lasso:
        if self.lambda_glasso == 0:
            mu_estimate = mu_t / self.n_obs
            sigma_estimate = (sigma_t / self.n_obs) - np.outer(mu_estimate, mu_estimate)
            return mu_estimate, sigma_estimate
        else:
            sigma_t_plus_one, _ = graphical_lasso(sigma_t,
                                                  self.lambda_glasso,
                                                  mode=self.solver,
                                                  max_iter=100,
                                                  cov_init=sigma_t,
                                                  eps=self.eps_glasso)

            return mu_t, sigma_t_plus_one

    def calculate_sufficient_statistics(self, mu, sigma):

        total_cov_estimate = np.zeros((self.n_dims, self.n_dims))
        total_mu_estimate = np.zeros(self.n_dims)
        for k, obs in enumerate(self.data):
            obs_is_nan = np.isnan(obs)
            obs_existing = obs[~obs_is_nan]
            mu_existing = mu[~obs_is_nan]
            mu_missing = mu[obs_is_nan]
            cov_estimate = np.zeros((self.n_dims, self.n_dims))
            mu_estimate = np.zeros(self.n_dims)

            sigma_existing_existing = sigma[np.ix_(~obs_is_nan, ~obs_is_nan)]
            sigma_missing_missing = sigma[np.ix_(obs_is_nan, obs_is_nan)]
            sigma_missing_existing = sigma[np.ix_(obs_is_nan, ~obs_is_nan)]
            inv_sigma_existing_existing = np.linalg.inv(sigma_existing_existing)
            expectation_hidden_given_visible = mu_missing + sigma_missing_existing @ inv_sigma_existing_existing @ \
                                               (obs_existing - mu_existing)

            # Calculate expectation of mu_i:
            mu_estimate[~obs_is_nan] = obs_existing
            mu_estimate[obs_is_nan] = expectation_hidden_given_visible

            covariance_hidden_given_visible = sigma_missing_missing \
                                              - sigma_missing_existing \
                                              @ inv_sigma_existing_existing \
                                              @ sigma_missing_existing.T

            # Calculate the outer product of the existing variables:
            cov_estimate[np.ix_(~obs_is_nan, ~obs_is_nan)] = np.outer(obs_existing, obs_existing)
            cov_estimate[np.ix_(obs_is_nan, obs_is_nan)] = covariance_hidden_given_visible \
                                                           + np.outer(expectation_hidden_given_visible,
                                                                      expectation_hidden_given_visible)
            cov_estimate[np.ix_(obs_is_nan, ~obs_is_nan)] = np.outer(expectation_hidden_given_visible, obs_existing)
            cov_estimate[np.ix_(~obs_is_nan, obs_is_nan)] = np.transpose(cov_estimate[np.ix_(obs_is_nan, ~obs_is_nan)])

            total_cov_estimate = total_cov_estimate + cov_estimate
            total_mu_estimate = total_mu_estimate + mu_estimate
        return total_mu_estimate, total_cov_estimate

    def run_em_algo(self):
        """
        :param self:
        :return:
        """
        current_diff = np.inf
        sigma_t = self.sigma_init.copy()
        mu = np.zeros(self.n_dims)
        diffs = []
        tries = 0
        sigmas = []
        while current_diff > self.eps:

            estimated_mu, estimated_sigma = self.e_step(mu, sigma_t)
            mu, sigma_t_plus_one = self.m_step(estimated_mu, estimated_sigma)
            if self.calculate_log_likelihood:
                log_score = self.log_likelihood(sigma=sigma_t_plus_one, mu=mu)
                self.log_likelihoods.append(log_score)
                print("The current log-likelihood is proportional to " + str(log_score))
                if len(self.log_likelihoods) >= 2:
                    diff_likelihoods = self.log_likelihoods[-2] - self.log_likelihoods[-1]
                    if diff_likelihoods > 0:
                        print("The log-likelihood is decreasing, the difference is: " + str(diff_likelihoods))
            sigmas.append(sigma_t_plus_one)
            if np.all(np.abs(sigma_t) > 0):
                current_diff = np.max(np.abs((sigma_t_plus_one - sigma_t) / sigma_t))
            else:
                current_diff = np.inf
            diffs.append(current_diff)
            sigma_t = sigma_t_plus_one.copy()
            tries = tries + 1
            if tries >= self.max_iter:
                break
        self.estimated_sparse_cov = sigma_t
        self.log_likelihood_result = self.log_likelihood(sigma=sigma_t)
        return sigma_t

    def log_likelihood(self, mu=None, sigma=None, data_to_score=None, penalized=True):
        """

        :return:
        """

        if sigma is None and self.estimated_sparse_cov is not None:
            sigma = self.estimated_sparse_cov
        elif sigma is None and self.estimated_sparse_cov is None:
            raise Exception("You have to provide a covariance matrix!")
        if mu is None:
            mu = np.zeros(self.n_dims)

        if data_to_score is None:
            data_to_score = self.data
            n_obs = self.n_obs
            missing_values = self.missing_values
        else:
            n_obs = data_to_score.shape[0]
            missing_values = np.ma.masked_invalid(data_to_score).mask
        existing_indices = np.argwhere(missing_values[:] == False)

        existing_values = []

        for k in range(n_obs):
            existing_values.append([row[1] for row in existing_indices if row[0] == k])
        # Here we have to find a better solution and just calculate the inverse of matrices once!
        # Also note, that we have to take the [0][0] as a matrix 1x1 matrix is returned:
        # Here we have to set "== True " due to numpy conventions
        score = 0
        for k, obs in enumerate(data_to_score):
            obs_is_nan = np.isnan(obs)
            obs_existing = obs[~obs_is_nan]
            sigma_existing = sigma[np.ix_(~obs_is_nan, ~obs_is_nan)]
            mu_existing = mu[~obs_is_nan]
            current_score = -0.5 * float(np.log(np.linalg.det(sigma_existing)) - 0.5 * (obs_existing - mu_existing).dot(
                np.linalg.inv(sigma_existing)).dot(
                obs_existing - mu_existing))

            score = current_score + score

        if penalized:
            return score - self.lambda_glasso * np.linalg.norm(sigma, 1)
        if not penalized:
            return score


def likelihood_score(data_to_score, estimated_cov, mu=None):
    """
    # TODO: Documentation!
    :param mu:
    :param data_to_score:
    :param estimated_cov:
    :return:
    """
    if mu is None:
        mu = np.zeros(estimated_cov.shape[0])
    sigma = estimated_cov
    # Here we have to find a better solution and just calculate the inverse of matrices once!
    # As we want to score the fit, we do not take the penalty here:
    score = 0
    for k, obs in enumerate(data_to_score):
        obs_is_nan = np.isnan(obs)
        obs_existing = obs[~obs_is_nan]
        mu_existing = mu[~obs_is_nan]
        sigma_existing = sigma[np.ix_(~obs_is_nan, ~obs_is_nan)]
        current_score = -0.5 * float(np.log(np.linalg.det(sigma_existing)) + 0.5 * (obs_existing - mu_existing).dot(
            np.linalg.inv(sigma_existing)).dot(
            obs_existing - mu_existing))

        score = current_score + score

    return score


def make_k_fold_evaluation_em_glasso(pd_dataframe,
                                     alpha,
                                     n_splits=5,
                                     init_shrinkage=0,
                                     random_state=None,
                                     return_all_scores=False,
                                     **kwargs):
    """
    TODO: DOC
    :param return_all_scores:
    :param random_state:
    :param init_shrinkage:
    :param pd_dataframe:
    :param alpha:
    :param n_splits:
    :return:
    """
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    score_estimates = []
    for train_index, test_index in k_fold.split(pd_dataframe):
        X_train = pd_dataframe.iloc[train_index, :]
        X_train_cov = shrunk_covariance(X_train.cov(), shrinkage=init_shrinkage)
        X_test = pd_dataframe.iloc[train_index, :]
        X_train_np = X_train.to_numpy()

        em_glasso = EMGlasso(X_train_np, X_train_cov, lambda_glasso=alpha, **kwargs)

        print("We are trying " + str(em_glasso.lambda_glasso))
        em_glasso.run_em_algo()
        mu_estimated = X_train.mean(axis=0).to_numpy()
        score_current_run = em_glasso.log_likelihood(mu=mu_estimated, data_to_score=X_test.to_numpy(), penalized=False)
        print("Score of current run " + str(score_current_run))

        score_estimates.append(float(score_current_run))

    if return_all_scores:
        return score_estimates, np.mean(score_estimates)
    else:
        return np.mean(score_estimates)


def predict_target_values(np_array_for_prediction, index_target, sigma):
    missing_entries = np.isnan(np_array_for_prediction)
    mask = [False if k in index_target else True for k, element in enumerate(np_array_for_prediction)]
    existing_and_not_in_target_index = (~missing_entries) & mask

    y_indices = [not element for element in mask]
    # sigma_(2,2) ** (-1)
    sigma_inv_existing = np.linalg.inv(
        sigma[np.ix_(existing_and_not_in_target_index, existing_and_not_in_target_index)])
    # sigma_(2,1)
    sigma_y_x_existing = sigma[np.ix_(existing_and_not_in_target_index, y_indices)]
    return (sigma_inv_existing @ sigma_y_x_existing).T @ np_array_for_prediction[existing_and_not_in_target_index]


class EMCorrelation(EMGlasso):
    def __init__(self, data, sigma_init,
                 eps=0.01,
                 calculate_log_likelihood=False,
                 max_iter=20,
                 eps_correlation_finder=0.001):
        EMGlasso.__init__(self, data, sigma_init,
                          lambda_glasso=0.0,
                          eps=eps,
                          calculate_log_likelihood=calculate_log_likelihood,
                          solver=None,
                          max_iter=max_iter,
                          eps_glasso=None,
                          )
        self.eps_correlation_finder = eps_correlation_finder

    def m_step(self, mu_t, sigma_t):
        """
        Here, we search for the maximizing correlation matrix via a gradient descent approach.
        :param mu_t:
        :param sigma_t:
        :return:
        """
        mu_estimate = mu_t / self.n_obs
        cov_estimate = (sigma_t / self.n_obs) - np.outer(mu_estimate, mu_estimate)
        corr_finder = OptimalCorrelationFinder(cov_estimate, self.eps_correlation_finder, 0.3, 1000)
        optimal_correlation = corr_finder.maximize_correlation()
        return mu_estimate, optimal_correlation
