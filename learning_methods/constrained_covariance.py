import numpy as np


class ConstrainedCovarianceEstimator:
    """
    This class solves
    argmax_\theta log det(\theta) - trace(S \theta)
    under the constraint that \theta_ij = 0 for some pairs i,j.
    The constraints are given by p x p matrix, where 1s indicate that the entries are constraint to
    be 0.
    We solve this using Lagrange, see https://hastie.su.domains/TALKS/wald_III.pdf.
    Why is this algorithm guaranteed to converge? The similar Graphical Lasso is not guaranteed to converge if
    starting point is chosen incorrectly (page 10 of https://hastie.su.domains/Papers/glasso_revisit_trevor_arxiv.pdf).
    """

    def __init__(self, empirical_covariance, constraints,
                 init_estimate=None, eps=.0001):
        """
        :param empirical_covariance:
        :param constraints:
        :param init_estimate:
        """
        self.empirical_covariance = empirical_covariance.copy()
        if np.trace(constraints) != 0:
            raise Exception("The diagonal elements can not be restricted to 0!")
        elif sorted(np.unique(constraints)) != [0., 1.]:
            raise Exception("The constraint matrix must have elements in set (0,1)!")
        else:
            self.constraints = constraints
        if self.empirical_covariance.shape != self.constraints.shape:
            raise Exception("Shapes of empirical covariance and constraint matrix must align")
        if self.empirical_covariance.shape[0] != self.empirical_covariance.shape[1]:
            raise Exception("Empirical covariance matrix must be symmetric!")
        self.p = self.empirical_covariance.shape[0]
        if init_estimate is not None:
            self.W = init_estimate
        else:
            self.W = self.empirical_covariance
        self.edges_exist = np.where(self.constraints == 1)
        self.eps = eps
        self.betas = np.zeros((self.p - 1, self.p))
        self.theta = np.zeros((self.p, self.p))
        self.n_iter = 0

    def cycle_step(self, k):
        if k >= self.p:
            raise Exception("Cycle at most until p = {}".format(self.p + 1))
        # TODO: Take out of this function and initialize that somewhere else.
        constraints_curr = np.where(self.constraints[:, k] == 0)[0]
        minus_k = [l for l in range(self.p) if l != k]
        W_minus_p_minus_p = self.W[np.ix_(minus_k, minus_k)]
        minus_k_and_no_zero_constraint = [l for l in range(self.p) if l != k and l in constraints_curr]
        s_star = self.empirical_covariance[minus_k_and_no_zero_constraint, k]
        # The todo-task is relavant until at least this point.
        W_star = self.W[np.ix_(minus_k_and_no_zero_constraint, minus_k_and_no_zero_constraint)]
        beta_star = np.linalg.solve(W_star, s_star)
        beta = np.zeros(self.p)
        beta[np.ix_(minus_k_and_no_zero_constraint)] = beta_star
        w_minus_k_k_new = W_minus_p_minus_p @ beta[minus_k]
        self.W[np.ix_(minus_k), k] = w_minus_k_k_new
        self.betas[:, k] = beta[minus_k]
        return self.W

    def run_algorithm(self):
        current_eps = np.inf
        while current_eps > self.eps:
            W_old = self.W.copy()
            for k in range(self.p):
                self.cycle_step(k)
            current_eps = np.linalg.norm(self.W - W_old)
            #print("The current eps is {}".format(current_eps))
            #print("The current covariance matrix is {}".format(self.W))
            self.n_iter += 1

        for k in range(self.p):
            minus_k = [l for l in range(self.p) if l != k]
            theta_k_k = (self.empirical_covariance[k, k] - self.W[minus_k, k] @ (self.betas[:, k])) ** (-1)
            self.theta[minus_k, k] = - self.betas[:, k] * theta_k_k
            self.theta[k, k] = theta_k_k

        return self.W, self.theta

