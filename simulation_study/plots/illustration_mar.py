import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from copula_modelling.copula import GaussianCopula
from copula_modelling.marginal_models.mixture import GaussianMixture
from learning_methods.EMGlasso import EMGlasso
from learning_methods.nonparanormal_estimation import EmpiricalMarginalCDFEstimator
from learning_methods.nonparanormal_estimation import sample_copula
from simulation_study.evaluate.base_function import generate_data2d
from simulation_study.plots.plot_settings import tex_fonts
from simulation_study.plots.utils import get_right_palette_for_subset

np.random.seed(93)

method = "shrinkage"
savefig = True
distributions = [stats.chi2(6), stats.chi2(7)]
n_dim = len(distributions)
rho = 0.5
n_rows = 200
p_mcar = 0.1
cov = np.array([[1, rho], [rho, 1]])

n_sample_distplot = 10000

regression_parameter = 2
shift = 0

plt.style.use('seaborn-paper')
plt.rcParams.update(tex_fonts)

transformed_data_wo_missing, transformed_data_missing, data_gaussian_missing, data_gaussian = generate_data2d(cov, n_rows, distributions,
                                                                                                              regression_parameter, shift,
                                                                                                              transformed_data_wo_missing=True)
em_algo_start = np.eye(n_dim)
verbose = False

num_mixtures = 20
eps = 0.003
# M = np.concatenate((np.array(20 * [50]), np.array(5 * [1000])))
M = np.array(20 * [50])


# Proposed Algorithm
stds = 1.06 * num_mixtures ** (-1 / 5) * np.nanstd(transformed_data_missing, axis=0)

gauss_mixtures = []
for k in range(data_gaussian.shape[1]):
    curr_gaussian_mixture = GaussianMixture(np.zeros(num_mixtures), stds[k], verbose=verbose)
    params_result = curr_gaussian_mixture.maximize_likelihood(transformed_data_missing[:, k])
    curr_gaussian_mixture.set_params(params_result)
    gauss_mixtures.append(curr_gaussian_mixture)

cop_model = GaussianCopula(transformed_data_missing,
                               em_algo_start,
                               M,
                               marginal_distributions=gauss_mixtures,
                               verbose=verbose,
                               eps=eps
                               )
estimated_sigma, estimated_theta = cop_model.run_em_algo()
copula_sample = cop_model.sample(n_sample_distplot)
# Apply SCOPE
normalized_data = np.zeros_like(transformed_data_missing)

for i in range(transformed_data_missing.shape[1]):
    current_col = transformed_data_missing[:, i]
    normalized_data[:, i] = EmpiricalMarginalCDFEstimator(current_col, method=method).gaussianize_data()

# Apply the EM algorithm for SCOPE:

cov_scope = EMGlasso(normalized_data,
                     em_algo_start,
                     lambda_glasso=0,
                     calculate_log_likelihood=False,
                     max_iter=1000,
                     eps=10 ** (-5)
                     ).run_em_algo()

rescaling_matrix = np.outer(1 / np.sqrt(np.diagonal(cov_scope)),
                            1 / np.sqrt(np.diagonal(cov_scope))
                            )

cor_scope = cov_scope * rescaling_matrix
sample_scope = sample_copula(transformed_data_missing, cor_scope, n_sample_distplot,
                             marginal_cdf_method=method)

df_marginals_hoff = pd.read_csv("C:/Users/Q508552/Desktop/Copula/SimulationResults/Entropy/2D/marginals.csv")

# Plot the cumulative distribution estimates:
plt.figure(figsize=(16, 9))
plt.style.use('seaborn-paper')
plt.rcParams.update(tex_fonts)
for k in range(n_dim):
    plt.subplot(1, 2, k + 1)
    original_data = transformed_data_wo_missing[:, k]
    x_min = np.nanquantile(original_data, 0.001)
    x_max = np.nanquantile(original_data, 0.999)
    grid = np.linspace(x_min, x_max, 100)
    widehat_scope = r'$\widehat{F}_1^{SCOPE}$' if k == 0 else r'$\widehat{F}^{SCOPE}_2$'
    widehat_mcmc = r'$\widehat{F}_1^{MCMC}$' if k == 0 else r'$\widehat{F}^{MCMC}_2$'
    widehat_em = r'$\widehat{F}_1^{EM}$' if k == 0 else r'$\widehat{F}^{EM}_2$'
    curr_marginal_mcmc = df_marginals_hoff["first"] if k == 0 else df_marginals_hoff["second"]
    color_scope = get_right_palette_for_subset([widehat_scope])[0]
    color_em = get_right_palette_for_subset([widehat_em])[0]
    color_mcmc = get_right_palette_for_subset([widehat_mcmc])[0]
    cop_model.marginal_distributions[k].plot_cdf(grid=grid,
                                                 label=widehat_em,
                                                 color=color_em)
    sns.ecdfplot(pd.DataFrame(transformed_data_missing[:, k], columns=[widehat_scope]),
                 label=widehat_scope,
                 color=color_scope)

    sns.ecdfplot(curr_marginal_mcmc, label=widehat_mcmc, color=color_mcmc)
    #sns.ecdfplot(original_data, label="ECDF")
    linspace = np.linspace(np.min(transformed_data_wo_missing[:, k]), np.max(transformed_data_wo_missing[:, k]), 1000)
    sns.lineplot(linspace, distributions[k].cdf(linspace),
                 label=r"$F_{}$".format(k+1),
                 color="green")

    plt.legend()
    if savefig:
        plt.savefig("C:\\Users\\Q508552\\Documents\\LaTex\\Paper\\GaussianCopulaMAR\\Entropy\\Plots\\results_marginals_example.pdf",
                    format="jpg")
plt.show()
