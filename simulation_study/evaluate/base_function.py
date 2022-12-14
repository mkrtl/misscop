from dcor import energy_distance
from scipy.stats import cramervonmises_2samp
import time

from simulation_study.evaluate.parameters import *
from copula_modelling.copula import GaussianCopula
from learning_methods.nonparanormal_estimation import *
from learning_methods.EMGlasso import EMGlasso
from copula_modelling.marginal_models.mixture import GaussianMixture


def run_simulation(current_seed,
                   l=0,
                   num_mixtures=num_mixtures,
                   M=M,
                   n_sample=n_sample,
                   eps=eps,
                   method=method,
                   cov=cov,
                   n_rows=n_rows,
                   verbose=False,
                   regression_parameter=regression_parameter,
                   shift=shift,
                   distributions=distributions,
                   known_precision_structure=None):
    np.random.seed(current_seed)
    print("Simulation {} started with seed {}".format(l, current_seed))
    n_dim = cov.shape[0]
    if n_dim not in [2, 3]:
        raise Exception("Function only implemented for two- or three-dimensional case.")

    if n_dim == 2:
        transformed_data_missing, data_gaussian_missing, data_gaussian = generate_data2d(cov, n_rows, distributions,
                                                                                         regression_parameter, shift)

    else:
        transformed_data_missing, data_gaussian_missing, data_gaussian = generate_data3d(cov, n_rows, distributions,
                                                                                         regression_parameter, shift)

    share_missing_values = np.sum(np.isnan(transformed_data_missing), axis=0)

    result_dict = dict()
    result_dict["seed"] = current_seed
    result_dict["share_missing_columns"] = share_missing_values / n_rows

    # All algorithms start agnostic and assume independent components:
    em_algo_start = np.eye(n_dim)

    # Apply the proposed algorithm
    """
    Applying one kind of rule of thumb as described here:
    https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
    """
    stds = 1.06 * num_mixtures ** (-1 / 5) * np.nanstd(transformed_data_missing, axis=0)

    gauss_mixtures = []
    for k in range(data_gaussian.shape[1]):
        curr_gaussian_mixture = GaussianMixture(np.zeros(num_mixtures), stds[k], verbose=verbose)
        params_result = curr_gaussian_mixture.maximize_likelihood(transformed_data_missing[:, k])
        curr_gaussian_mixture.set_params(params_result)
        gauss_mixtures.append(curr_gaussian_mixture)

    now = time.time()
    cop_model = GaussianCopula(transformed_data_missing,
                               em_algo_start,
                               M,
                               marginal_distributions=gauss_mixtures,
                               verbose=verbose,
                               eps=eps
                               )
    estimated_sigma, estimated_theta = cop_model.run_em_algo()
    result_dict["time_estimator"] = time.time() - now
    copula_sample = cop_model.sample(n_sample)
    result_dict["cov_estimator"] = estimated_sigma
    result_dict["theta_estimator"] = estimated_theta
    # If there is a known correlation structure:
    if known_precision_structure is not None:
        now = time.time()
        cop_model_known_precision_structure = GaussianCopula(transformed_data_missing,
                                                             em_algo_start,
                                                             M,
                                                             marginal_distributions=gauss_mixtures,
                                                             verbose=verbose,
                                                             eps=eps,
                                                             sigma_inverse_constraint=known_precision_structure,
                                                             )

        estimated_sigma_known_precision, \
        estimated_theta_known_precision = cop_model_known_precision_structure.run_em_algo()
        result_dict["time_known_precision"] = time.time() - now
        copula_sample_known_precision = cop_model_known_precision_structure.sample(n_sample)
        result_dict["theta_estimator_known_precision"] = estimated_theta_known_precision
        result_dict["estimated_sigma_known_precision"] = estimated_sigma_known_precision

    # If the marginals are already known, then we can fit the EM algorithm for multivariate normal data using the
    # Gaussian data.
    now = time.time()
    cov_sigma_known_marginals = EMGlasso(data_gaussian_missing,
                                         em_algo_start,
                                         lambda_glasso=0,
                                         calculate_log_likelihood=False,
                                         max_iter=1000,
                                         eps=10 ** (-5),
                                        ).run_em_algo()
    result_dict["time_known_marginals"] = time.time() - now
    rescaling_matrix_known_marginals = np.outer(1 / np.sqrt(np.diagonal(cov_sigma_known_marginals)),
                                                1 / np.sqrt(np.diagonal(cov_sigma_known_marginals))
                                                )
    cor_sigma_known_marginals = cov_sigma_known_marginals * rescaling_matrix_known_marginals
    gaussian_sample_estimated_cor_known_marginals = stats.multivariate_normal(cov=cor_sigma_known_marginals) \
        .rvs(n_sample)
    sample_known_marginals = np.zeros_like(gaussian_sample_estimated_cor_known_marginals)
    for col in range(gaussian_sample_estimated_cor_known_marginals.shape[1]):
        sample_known_marginals[:, col] = distributions[col].ppf(
            stats.norm.cdf(gaussian_sample_estimated_cor_known_marginals[:, col])
        )
    result_dict["cov_known_marginals"] = cor_sigma_known_marginals

    # Apply SCOPE
    normalized_data = np.zeros_like(transformed_data_missing)

    for i in range(transformed_data_missing.shape[1]):
        current_col = transformed_data_missing[:, i]
        normalized_data[:, i] = EmpiricalMarginalCDFEstimator(current_col, method=method).gaussianize_data()

    # Apply the EM algorithm for SCOPE:
    now = time.time()
    cov_scope = EMGlasso(normalized_data,
                         em_algo_start,
                         lambda_glasso=0,
                         calculate_log_likelihood=False,
                         max_iter=1000,
                         eps=10 ** (-5)
                         ).run_em_algo()
    result_dict["time_scope"] = time.time() - now

    rescaling_matrix = np.outer(1 / np.sqrt(np.diagonal(cov_scope)),
                                1 / np.sqrt(np.diagonal(cov_scope))
                                )

    cor_scope = cov_scope * rescaling_matrix

    result_dict["cor_scope"] = cor_scope

    sample_scope = sample_copula(transformed_data_missing, cor_scope, n_sample,
                                 marginal_cdf_method=method)

    # Sample from the real underlying distribution:
    data_normal_sample = stats.multivariate_normal(cov=cov).rvs(n_sample)
    sample_data = np.zeros_like(data_normal_sample)
    for i in range(len(distributions)):
        sample_data[:, i] = distributions[i].ppf(stats.norm.cdf(data_normal_sample[:, i]))

    # Do the evaluation:
    test_stat_estimator = evaluate_samples(sample_data, copula_sample)
    test_stat_scope = evaluate_samples(sample_data, sample_scope)
    test_stat_known_marginals = evaluate_samples(sample_data, sample_known_marginals)

    result_dict["test_stat_scope"] = test_stat_scope
    result_dict["test_stat_estimator"] = test_stat_estimator
    result_dict["test_stat_known_marginals"] = test_stat_known_marginals

    if known_precision_structure is not None:
        test_stat_known_precision = evaluate_samples(sample_data, copula_sample_known_precision)
        result_dict["test_stat_known_precision"] = test_stat_known_precision
        print("Assuming the precision structure is known:")
        print("Test statistic assuming known precision {}".format(test_stat_known_precision))
        print("Correlation {}".format(estimated_sigma_known_precision))

    print("Run number {} finished.".format(l))
    print("########################################################")
    print("SCOPE Estimator")
    print("Test statistic {}".format(test_stat_scope))
    print("Correlation {}".format(cor_scope))
    print("-------------------------------------------------------")
    print("Own estimator")
    print("Test statistic {}".format(test_stat_estimator))
    print("Correlation {}".format(estimated_sigma))
    print("-------------------------------------------------------")
    print("Assuming the marginals are known:")
    print("Test statistic assuming known marginals {}".format(test_stat_known_marginals))
    print("Correlation {}".format(cor_sigma_known_marginals))
    print("---------------------------------------------------------------")
    print("Share of missing values:")
    print("By columnmn: {}".format(share_missing_values / n_rows))
    print("################################################################")
    return result_dict


def generate_data2d(cov, n_rows, distributions, regression_parameter, shift, transformed_data_wo_missing=False):
    """
    Function that generates data set and applies missing mechanism
    :return:
    """
    data_gaussian, data_gaussian_missing = _generate_gaussian_data_and_apply_mcar(cov, n_rows)
    # Enforce MAR mechanism:
    for k, row in enumerate(data_gaussian_missing):
        p_missing = stats.uniform().rvs()
        # Assume that the missingness follows the logistic regression model:
        if (1 / (1 + np.exp(-(row[0] * regression_parameter + shift)))) > p_missing:
            data_gaussian_missing[k, 1] = np.nan
    transformed_data_missing = _transform_marginals(data_gaussian_missing, distributions)
    transformed_data = _transform_marginals(data_gaussian, distributions)
    if transformed_data_wo_missing:
        return transformed_data, transformed_data_missing, data_gaussian_missing, data_gaussian
    else:
        return transformed_data_missing, data_gaussian_missing, data_gaussian


def _generate_gaussian_data_and_apply_mcar(cov, n_rows):
    data_gaussian = stats.multivariate_normal(cov=cov).rvs(size=n_rows)
    data_gaussian_missing = data_gaussian.copy()
    mcar_indicator = np.reshape(stats.uniform.rvs(size=data_gaussian.size),
                                (data_gaussian.shape[0], data_gaussian.shape[1]))
    # Make some entries missing completely at random:
    data_gaussian_missing[mcar_indicator > (1.0 - p_mcar)] = np.nan
    return data_gaussian, data_gaussian_missing


def _transform_marginals(data_gaussian_missing, distributions):
    # Transform the data, such that it has the given marginal distributions:
    transformed_data_missing = np.zeros_like(data_gaussian_missing)
    for i in range(len(distributions)):
        transformed_data_missing[:, i] = distributions[i].ppf(stats.norm.cdf(data_gaussian_missing[:, i]))
    return transformed_data_missing[~np.all(np.isnan(transformed_data_missing), axis=1), :]


def generate_data3d(cov, n_rows, distributions, regression_parameter, shift, transformed_data_wo_missing=False):
    """
    Function that generates data set and applies missing mechanism.
    Here the regression parameter is two-dimensional.
    :return:
    """
    data_gaussian, data_gaussian_missing = _generate_gaussian_data_and_apply_mcar(cov, n_rows)
    # Enforce MAR mechanism:
    for k, row in enumerate(data_gaussian_missing):
        p_missing = stats.uniform().rvs()
        # Assume that the missingness follows the logistic regression model:
        if not (np.isnan(row[0]) or np.isnan(row[1])):
            if (1 / (1 + np.exp(-(row[[0, 1]].T @ regression_parameter + shift)))) > p_missing:
                data_gaussian_missing[k, 2] = np.nan
        elif not (np.isnan(row[0]) or np.isnan(row[2])):
            if (1 / (1 + np.exp(-(row[[0, 2]].T @ regression_parameter + shift)))) > p_missing:
                data_gaussian_missing[k, 1] = np.nan
    transformed_data_missing = _transform_marginals(data_gaussian_missing, distributions)
    transformed_data = _transform_marginals(data_gaussian, distributions)
    if transformed_data_wo_missing:
        return transformed_data, transformed_data_missing, data_gaussian_missing, data_gaussian
    else:
        return transformed_data_missing, data_gaussian_missing, data_gaussian


def evaluate_samples(copula_sample_true, samples):
    p = copula_sample_true.shape[1]
    energy_distance_joint = energy_distance(copula_sample_true, samples)
    energy_distances_marginals = np.zeros(p)
    for k in range(p):
        energy_distances_marginals[k] = cramervonmises_2samp(samples[:, k], copula_sample_true[:, k],
                                                             method="asymptotic").statistic
    return dict(joint_distance=energy_distance_joint, distance_marginals=energy_distances_marginals)


def generate_only_datasets(cov,
                           n_rows,
                           distributions,
                           regression_parameter,
                           shift,
                           seed,
                           transformed_data_wo_missing=False):
    np.random.seed(seed)
    if cov.shape[0] == 3:
        data = generate_data3d(cov,
                                  n_rows,
                                  distributions,
                                  regression_parameter,
                                  shift,
                                  transformed_data_wo_missing=transformed_data_wo_missing)[0]
    elif cov.shape[0] == 2:
        data = generate_data2d(cov,
                               n_rows,
                               distributions,
                               regression_parameter,
                               shift,
                               transformed_data_wo_missing=transformed_data_wo_missing)[0]
    return data
