from scipy import stats
import numpy as np

distributions = [stats.chi2(6), stats.chi2(7)]
n_dim = len(distributions)
num_mixtures = 15
rho = .1
n_rows = 100
p_mcar = 0.1
shift = 0
regression_parameter = 2
cov = np.array([[1, rho], [rho, 1]])
method = "shrinkage"

eps = 0.003
M = np.concatenate((np.array(20 * [50]), np.array(5 * [1000])))
n_sample = 10000
