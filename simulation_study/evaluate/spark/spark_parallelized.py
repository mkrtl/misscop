# Databricks notebook source
from functools import partial
import numpy as np
from scipy import stats

from simulation_study.evaluate.base_function import run_simulation

# COMMAND ----------

N = 100
seed_max = 2 ** 32 - 1

rng = np.random.default_rng(93)
seeds_to_use = rng.integers(0, seed_max, N)
object_to_parallelize = [[seed, i] for i, seed in enumerate(seeds_to_use)]

distributions = [stats.chi2(6), stats.chi2(7), stats.chi2(5)]
n_dim = len(distributions)
num_mixtures = 15
n_rows = 50
p_mcar = 0.1
shift = 0
method = "shrinkage"
regression_parameter = np.array([2, 2])
precision = np.array([[1, 0., 0.5], [0., 1., .5], [.5, .5, 1]])
cov = np.linalg.inv(precision)
rescaling_matrix = 1 / np.sqrt(np.outer(np.diagonal(cov), np.diagonal(cov)))
cov = np.multiply(cov, rescaling_matrix)
known_precision_structure = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])


eps = 0.003
M = np.concatenate((np.array(20 * [50]), np.array(5 * [1000])))
n_sample = 10000

run_simulation_only_l_and_seed = partial(run_simulation,
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
                                         known_precision_structure=known_precision_structure
                                         )

# COMMAND ----------

mapped_seeds = sc.parallelize(object_to_parallelize, N)

# COMMAND ----------

return_object = mapped_seeds.map(lambda a: run_simulation_only_l_and_seed(*a), preservesPartitioning=True)
results_list = return_object.collect()

# COMMAND ----------

'dbfs:/FileStore/simulation_results_entropy/rho{}shift{}reg{}N{}.csv'.format(str(cov).replace(".", ""), 
                                                                             shift, 
                                                                             str(regression_parameter).replace(",","").replace("[]", 
                                                                             N)

# COMMAND ----------

import pickle

# COMMAND ----------

results_list

# COMMAND ----------

np.mean([ent["test_stat_scope"]["joint_distance"] for ent in results_list])

# COMMAND ----------

np.mean([ent["test_stat_estimator"]["joint_distance"] for ent in results_list])

# COMMAND ----------

np.mean([ent["test_stat_known_precision"]["joint_distance"] for ent in results_list])

# COMMAND ----------

np.mean([ent["test_stat_scope"]["distance_marginals"][0] for ent in results_list])

# COMMAND ----------

np.mean([ent["test_stat_estimator"]["distance_marginals"][0] for ent in results_list])

# COMMAND ----------

np.mean([ent["test_stat_known_precision"]["distance_marginals"][0] for ent in results_list])

# COMMAND ----------

np.mean([np.linalg.norm(ent["cor_scope"] - cov) for ent in results_list])

# COMMAND ----------

np.mean([np.linalg.norm(ent["cov_estimator"] - cov) for ent in results_list])

# COMMAND ----------

np.mean([np.linalg.norm(ent["estimated_sigma_known_precision"] - cov) for ent in results_list])

# COMMAND ----------

np.mean([np.linalg.norm(ent["cov_known_marginals"] - cov) for ent in results_list])

# COMMAND ----------

with open('/dbfs/FileStore/simulation_results/first_run.pickle', 'wb') as handle:
    pickle.dump(results_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

# COMMAND ----------

with open('/dbfs/FileStore/simulation_results/first_run.pickle', 'rb') as handle:
  a = pickle.load(handle)
