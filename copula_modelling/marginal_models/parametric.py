from copula_modelling.marginal_models.base import MarginalModel
from scipy.stats import chi2
from scipy.special import digamma
import numpy as np
from scipy.optimize import approx_fprime


class ChiSquared(MarginalModel):

    def __init__(self, params, verbose=False):
        # params is only one value and corresponds to the degrees of freedom (df)
        super().__init__(params, verbose=verbose)
        self._initialize_class(params)
        self.name = "chi2"

    def _initialize_class(self, params):
        if not isinstance(params, (int, float)):
            if params.shape[0] > 1:
                raise ValueError("The number of parameters for chisquare has to be equal to 1.")
            else:
                self.df = params[0]
        else:
            self.df = params
        self.dist = chi2(self.df)

    def cdf(self, x):
        return self.dist.cdf(x)

    def cdf_inverse(self, probs):
        return self.dist.ppf(probs)

    def pdf(self, x):
        return self.dist.pdf(x)

    def pdf_derivative_mu(self, x):
        return ((np.log(x) - np.log(2)) / 2 - (1/2) * digamma(self.df/2)) * self.pdf(x)

    def cdf_derivative_mu(self, X, df=None):
        if df is None:
            df = self.df
        """

        # Lazy approach - maybe there is an explicit formula for this derivative as well
        epsilon = 0.001
        derivatives = np.zeros(X.shape[0])
        for k, x in enumerate(X):
            # make things faster as there are always the same values coming in:
            if k > 0 and X[k] == X[k-1]:
                derivatives[k] = derivatives[k-1]
            else:
                fct = lambda d: chi2(d).cdf(x)
                #partial_cdf_df = partial(fct, x)
                derivatives[k] = approx_fprime(df, fct, epsilon)[0]
        """
        # See here: https://functions.wolfram.com/GammaBetaErf/GammaRegularized/introductions/Gammas/ShowAll.html
        # https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives
        #derivatives = -np.exp(-X / 2) * (X / 2) ** (df / 2 - 1) / gamma(df/2)
        from scipy.special import gammainc
        epsilon = 0.001
        derivatives = np.zeros(X.shape[0])

        for k, x in enumerate(X):
            # make things faster as there are always the same values coming in:
            if k > 0 and X[k] == X[k - 1]:
                derivatives[k] = derivatives[k - 1]
            else:
                fct = lambda d: gammainc(d/2, x/2)
                # partial_cdf_df = partial(fct, x)
                derivatives[k] = approx_fprime(df, fct, epsilon)[0]
        return derivatives

    def sample(self, n):
        return self.dist.rvs(size=n)

    def set_params(self, params):
        self._initialize_class(params)
