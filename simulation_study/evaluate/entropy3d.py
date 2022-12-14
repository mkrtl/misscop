from simulation_study.evaluate.base_function import run_simulation
from simulation_study.evaluate.parameters import *

curr_seed = 1564422514
distributions = [stats.chi2(6), stats.chi2(7), stats.chi2(5)]
n_dim = len(distributions)
num_mixtures = 15
n_rows = 100
p_mcar = 0.1
shift = 0
regression_parameter = np.array([2, 2])
precision = np.array([[1, 0., 0.3], [0., 1., .5], [.3, .5, 1]])
cov = np.linalg.inv(precision)
rescaling_matrix = 1 / np.sqrt(np.outer(np.diagonal(cov), np.diagonal(cov)))
cov = np.multiply(cov, rescaling_matrix)

known_precision_structure = np.array([[0., 1., 0.], [1., 0., .0], [.0, .0, 0.]])

eps = 0.003
M = np.concatenate((np.array(20 * [50]), np.array(5 * [1000])))
n_sample = 10000


run_simulation(curr_seed,
               num_mixtures=num_mixtures,
               M=M,
               n_sample=n_sample,
               eps=eps,
               method=method,
               cov=cov,
               n_rows=n_rows,
               regression_parameter=regression_parameter,
               shift=shift,
               distributions=distributions,
               known_precision_structure=known_precision_structure,
               verbose=True)
