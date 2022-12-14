from __future__ import division

import numpy as np
from numpy import random
from scipy.spatial.distance import pdist, cdist
from scipy.stats import kstwobign, pearsonr, norm
from scipy.stats import genextreme


def ks2d2s(x1, y1, x2, y2, nboot=None, extra=False):
    '''Two-dimensional Kolmogorov-Smirnov test on two samples.
    Taken from: https://raw.githubusercontent.com/syrte/ndtest/master/ndtest.py
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.

    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic. Returned if keyword `extra` is True.

    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate, but it certainly implies that the two samples are not significantly different. (cf. Press 2007)

    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8

    '''
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            #ix1 = random.choice(n, n1, replace=True)
            #ix2 = random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    if extra:
        return p, D
    else:
        return p


def avgmaxdist(x1, y1, x2, y2):
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return max(dmin, dmax)


def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d


def estat2d(x1, y1, x2, y2, **kwds):
    return estat(np.c_[x1, y1], np.c_[x2, y2], **kwds)


def estat(x, y, nboot=1000, replace=False, method='log', fitting=False):
    '''
    Energy distance statistics test.
    Reference
    ---------
    Aslan, B, Zech, G (2005) Statistical energy as a tool for binning-free
      multivariate goodness-of-fit tests, two-sample comparison and unfolding.
      Nuc Instr and Meth in Phys Res A 537: 626-636
    Szekely, G, Rizzo, M (2014) Energy statistics: A class of statistics
      based on distances. J Stat Planning & Infer 143: 1249-1272
    Brian Lau, multdist, https://github.com/brian-lau/multdist

    '''
    n, N = len(x), len(x) + len(y)
    stack = np.vstack([x, y])
    stack = (stack - stack.mean(0)) / stack.std(0)
    if replace:
        rand = lambda x: random.randint(x, size=x)
    else:
        rand = random.permutation

    en = energy(stack[:n], stack[n:], method)
    en_boot = np.zeros(nboot, 'f')
    for i in range(nboot):
        idx = rand(N)
        en_boot[i] = energy(stack[idx[:n]], stack[idx[n:]], method)

    if fitting:
        param = genextreme.fit(en_boot)
        p = genextreme.sf(en, *param)
        return p, en, param
    else:
        p = (en_boot >= en).sum() / nboot
        return p, en, en_boot


def energy(x, y, method='log'):
    dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        pass
    else:
        raise ValueError
    z = dxy.sum() / (n * m) - dx.sum() / n**2 - dy.sum() / m**2
    # z = ((n*m)/(n+m)) * z # ref. SR
    return z


class Peacock3D:
    """
    Shamelessly copied from: https://rdrr.io/cran/Peacock.test/src/R/peacock3.R
    """
    def __init__(self, x, y):
        assert x.shape[1] == 3
        assert (x.shape == y.shape)
        self.n = x.shape[0]
        self.two_n = self.n * 2

        self.x = x
        self.y = y

        self.xy_one = np.concatenate((self.x[:, 0], self.y[:, 0]))
        self.xy_two = np.concatenate((self.x[:, 1], self.y[:, 1]))
        self.xy_three = np.concatenate((self.x[:, 2], self.y[:, 2]))

        self.order_xy_one = np.argsort(self.xy_one)
        self.order_xy_two = np.argsort(self.xy_two)
        self.order_xy_three = np.argsort(self.xy_three)

    def _increase_count(self, count, max_count, w):
        if w <= self.n - 1:
            count = count + 1
        else:
            count = count - 1
        return count, max(max_count, abs(count))

    def calculate_ks_statistic(self):
        max_hnnn = max_hnpn = max_hpnn = max_hppn = 0
        max_hnnp = max_hnpp = max_hpnp = max_hppp = 0

        for zu in self.xy_one[np.ix_(self.order_xy_one)]:
            for zv in self.xy_two[self.order_xy_two]:
                hnnn = 0
                hnpn = 0
                hpnn = 0
                hppn = 0

                t = 0
                while t < self.two_n:
                    w = self.order_xy_three[t]
                    if self.xy_one[w] <= zu:
                        if self.xy_two[w] <= zv:
                            hnnn, max_hnnn = self._increase_count(hnnn, max_hnnn, w)
                        else:
                            hnpn, max_hnpn = self._increase_count(hnpn, max_hnpn, w)
                    else:
                        if self.xy_two[w] <= zv:
                            hpnn, max_hpnn = self._increase_count(hpnn, max_hpnn, w)
                        else:
                            hppn, max_hppn = self._increase_count(hppn, max_hppn, w)

                    t += 1

                hnnp = 0
                hnpp = 0
                hpnp = 0
                hppp = 0

                t = self.two_n - 1

                while t > 0:
                    w = self.order_xy_three[t]
                    if self.xy_one[w] <= zu:
                        if self.xy_two[w] <= zv:
                            hnnp, max_hnnp = self._increase_count(hnnp, max_hnnp, w)
                        else:
                            hnpp, max_hnpp = self._increase_count(hnpp, max_hnpp, w)
                    else:
                        if self.xy_two[w] <= zv:
                            hpnp, max_hpnp = self._increase_count(hpnp, max_hpnp, w)
                        else:
                            hppp, max_hppp = self._increase_count(hppp, max_hppp, w)
                    t -= 1

        return max(max_hppp, max_hpnp, max_hnnn, max_hnpn, max_hnnp, max_hppn, max_hpnn, max_hnpp) / self.n


def kl_divergence(density_1, density_2, samples_density_1):
    """
    Monte Carlo approximation of the KL divergence between two density functions.
    :param density_1:
    :param density_2:
    :param samples_density_2:
    :return:
    """
    n = samples_density_1.shape[0]
    return np.sum(np.log(density_1(samples_density_1) / density_2(samples_density_1))) / n


