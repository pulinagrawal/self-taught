import numpy as np
from copy import copy
from scipy.stats import norm
from scipy.stats import hypergeom


def hypgeom_mean(n, K, N):
    return n * K / N


def hypgeom_var(n, K, N):
    return n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))


def hypgeom_pmf(k, n, K, N):
    dist = hypergeom(M=N, n=K, N=n)
    return dist.pmf(k)

def emp_p_value(x, m, s):
    return norm.sf(x, loc=m, scale=s)


def comb(s, n, l):
    if s == 0 and n == 0:
        print(l)
        return 1
    elif s != 0 and n == 0:
        return 0
    else:
        return sum([comb(s-i, n-1, copy(l+[i])) for i in range(s+1)])

''' Following code has been copied from 
http://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
'''

def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1,nobs+1)/float(nobs)

def fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False):
    '''pvalue correction for false discovery rate

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests. Both are
    available in the function multipletests, as method=`fdr_bh`, resp. `fdr_by`.

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : {'indep', 'negcorr')

    Returns
    -------
    rejected : array, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----

    If there is prior information on the fraction of true hypothesis, then alpha
    should be set to alpha * m/m_0 where m is the number of tests,
    given by the p-values, and m_0 is an estimate of the true hypothesis.
    (see Benjamini, Krieger and Yekuteli)

    The two-step method of Benjamini, Krieger and Yekutiel that estimates the number
    of false hypotheses will be available (soon).

    Method names can be abbreviated to first letter, 'i' or 'p' for fdr_bh and 'n' for
    fdr_by.



    '''
    pvals = np.asarray(pvals)

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1./np.arange(1, len(pvals_sorted)+1))   #corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
##    elif method in ['n', 'negcorr']:
##        cm = np.sum(np.arange(len(pvals)))
##        ecdffactor = ecdf(pvals_sorted)/cm
    else:
        raise ValueError('only indep and negcorr implemented')
    reject = pvals_sorted <= ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected>1] = 1
    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected


if __name__ == '__main__':

    print(comb(7, 4, []))
