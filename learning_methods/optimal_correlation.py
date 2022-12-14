import numpy as np
import logging

"""
n_steps = 10000

eps = 10 ** (-5)
beta = 0.3
"""


class OptimalCorrelationFinder(object):

    def __init__(self, empirical_covariance, eps, beta, n_steps,
                 verbose=False,
                 num_reduction_step_size=10):
        self.empirical_covariance = empirical_covariance
        self.eps = eps
        self.beta = beta
        self.n_steps = n_steps
        self.verbose = verbose
        self.n_dims = self.empirical_covariance.shape[0]
        self.rho = self.rho_start()
        self.non_diag_elements = np.ones_like(self.empirical_covariance) - np.eye(self.n_dims)
        self.all_rhos = [self.rho]
        self.num_reduction_step_size = num_reduction_step_size
        self.scores = []
        self.n_iter = 0

    def rho_start(self):
        sqrt_diag_elements = np.sqrt(np.linalg.inv(np.eye(self.n_dims) * self.empirical_covariance))
        return sqrt_diag_elements @ self.empirical_covariance @ sqrt_diag_elements

    def derivative(self):
        omega = np.linalg.inv(self.rho)
        derivative = omega - omega @ self.empirical_covariance @ omega
        if self.verbose:
            print("Current derivative: {}".format(derivative))
        return derivative

    def score(self, rho=None):
        if rho is None:
            rho = self.rho
        omega = np.linalg.inv(rho)
        return -np.log(np.linalg.det(omega)) + np.trace(omega @ self.empirical_covariance)

    def maximize_correlation(self):
        score_old = np.inf
        for step in range(1, self.n_steps):
            derivative = self.derivative()
            score = np.inf
            for l in range(self.num_reduction_step_size):
                alpha = self.beta ** l
                rho_test = self.rho - alpha * derivative * self.non_diag_elements
                # Check if eigenvalues are all positive, then continue, otherwise reduce alpha
                if np.all(np.linalg.eigvals(rho_test) > 0):
                    omega_test = np.linalg.inv(rho_test)
                    score = -np.log(np.linalg.det(omega_test)) + np.trace(omega_test @ self.empirical_covariance)
                    if self.verbose:
                        print("Current Score: {}".format(score))
                else:
                    logging.info("Encountered non-positive matrix {}".format(rho_test))
                if score < score_old:
                    break
            self.rho = rho_test
            self.all_rhos.append(self.rho)
            if score_old <= score:
                logging.warning("We could not improve score and the difference is still {}".format(score_old - score))
                logging.warning("Derivative: {}".format(derivative))
                logging.warning("Current correlation matrix: {}".format(self.rho))
            self.scores.append(score)
            score_old = score

            if np.sum(np.abs(derivative * self.non_diag_elements)) < self.eps:
                break
        self.n_iter = step
        return self.rho
