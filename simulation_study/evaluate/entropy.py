from simulation_study.evaluate.base_function import run_simulation
from simulation_study.evaluate.parameters import *


curr_seed = 1993
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


result = run_simulation(curr_seed,
               num_mixtures=num_mixtures,
               M=M,
               n_sample=n_sample,
               eps=eps,
               method=method,
               cov=cov,
               n_rows=n_rows)
