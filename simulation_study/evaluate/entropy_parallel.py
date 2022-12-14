import multiprocessing as mp

from simulation_study.evaluate.base_function import run_simulation
from simulation_study.evaluate.parameters import *
from functools import partial

N = 36
seed_max = 2 ** 32 - 1

rng = np.random.default_rng(93)
seeds_to_use = rng.integers(0, seed_max, N)

distributions = [stats.chi2(6), stats.chi2(7)]
n_dim = len(distributions)
num_mixtures = 15
rho = .5
n_rows = 100
p_mcar = 0.1
shift = 0
regression_parameter = 2
cov = np.array([[1, rho], [rho, 1]])
method = "shrinkage"

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
                                         verbose=False)


if __name__ == "__main__":
    pool = mp.Pool(processes=6)  # mp.cpu_count())
    try:
        results = pool.starmap(run_simulation_only_l_and_seed,
                               map(lambda a, b: (a, b), seeds_to_use, range(1, N + 1)))
        """
        results_df = pd.DataFrame.from_dict(results)
        plt.figure()
        plt.title("Test stats for rho = {}".format(rho))
        sns.boxplot(data=results_df[["test_stat_naive", "test_stat_estimator", "test_stat_two_step"]])
        plt.show()
        plt.figure()
        plt.title("Correlation estimators for rho = {}".format(rho))
        sns.boxplot(data=results_df[["rho_naive", "rho_estimator", "rho_two_step"]])
        plt.show()
        """
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
