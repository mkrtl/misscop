from simulation_study.evaluate.base_function import generate_only_datasets
from scipy import stats
import numpy as np

"""
N = 1000
seed_max = 2 ** 32 - 1

rng = np.random.default_rng(93)
seeds_to_use = rng.integers(0, seed_max, N)

distributions = [stats.chi2(6), stats.chi2(7), stats.chi2(5)]
n_dim = len(distributions)
num_mixtures = 15
n_rows = 50
p_mcar = 0.1
shift = -1
regression_parameter = np.array([2, 2])
precision = np.array([[1, 0., 0.5], [0., 1., .5], [.5, .5, 1]])
cov = np.linalg.inv(precision)
rescaling_matrix = 1 / np.sqrt(np.outer(np.diagonal(cov), np.diagonal(cov)))
cov = np.multiply(cov, rescaling_matrix)
"""

N = 1000
seed_max = 2 ** 32 - 1

rng = np.random.default_rng(93)
seeds_to_use = rng.integers(0, seed_max, N)

distributions = [stats.chi2(6), stats.chi2(7)]
rho = .1
n_rows = 100
p_mcar = 0.1
shift = 0
regression_parameter = 2
cov = np.array([[1, rho], [rho, 1]])

output_data = list()

for n in range(N):
    seed = seeds_to_use[n]
    output_data.append(generate_only_datasets(cov,
                                              n_rows,
                                              distributions,
                                              regression_parameter,
                                              shift,
                                              seed,
                                              transformed_data_wo_missing=False)
                       )

for k in range(len(output_data)):
    curr_dta = output_data[k]
    if k == 0:
        csv_data = np.c_[curr_dta, k * np.ones(curr_dta.shape[0])]
    else:
        curr_dta = np.c_[curr_dta, k * np.ones(curr_dta.shape[0])]
        csv_data = np.concatenate((csv_data, curr_dta))

np.savetxt("C:/Users/Q508552/Desktop/Copula/Data/{}d_rho{}_shift{}_regression{}_N_{}.csv".format(len(distributions),
                                                                                                 rho,
                                                                                                 shift,
                                                                                                 regression_parameter,
                                                                                                 n_rows), csv_data,
           delimiter=','
           )
