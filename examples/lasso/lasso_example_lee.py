import functools

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.tests.instance import gaussian_instance
from learn_selection.knockoffs import lasso_glmnet

from learn_selection.core import (infer_general_target,
                                  split_sampler,
                                  normal_sampler,
                                  logit_fit,
                                  probit_fit)

def simulate(n=200, p=50, s=5, signal=(2, 3), sigma=2, alpha=0.1):

    # description of statistical problem

    X, y, truth = gaussian_instance(n=n,
                                    p=p, 
                                    s=s,
                                    equicorrelated=False,
                                    rho=0.5, 
                                    sigma=sigma,
                                    signal=signal,
                                    random_signs=True,
                                    scale=False)[:3]

    dispersion = sigma**2

    S = X.T.dot(y)
    covS = dispersion * X.T.dot(X)
    smooth_sampler = normal_sampler(S, covS)
    splitting_sampler = split_sampler(X * y[:, None], covS)

    def meta_algorithm(X, XTXi, resid, sampler):

        S = sampler(scale=0.) # deterministic with scale=0
        ynew = X.dot(XTXi).dot(S) + resid # will be ok for n>p and non-degen X
        G = lasso_glmnet(X, ynew, *[None]*4)
        select = G.select()
        return set(list(select[0]))

    XTX = X.T.dot(X)
    XTXi = np.linalg.inv(XTX)
    resid = y - X.dot(XTXi.dot(X.T.dot(y)))
    dispersion = np.linalg.norm(resid)**2 / (n-p)
                         
    selection_algorithm = functools.partial(meta_algorithm, X, XTXi, resid)

    # run selection algorithm

    observed_set = selection_algorithm(splitting_sampler)

    # find the target, based on the observed outcome

    # we just take the first target  

    pivots, covered, lengths = [], [], []
    naive_pivots, naive_covered, naive_lengths =  [], [], []

    observed_list = sorted(observed_set)
    for idx in observed_list[:1]:
        print("variable: ", idx, "total selected: ", len(observed_set))
        true_target = truth[idx]

        linfunc = np.linalg.pinv(X[:,observed_list])[idx]
        observed_target = np.array([linfunc.dot(y)])
        cov_target = np.array([[np.linalg.norm(linfunc)**2 * dispersion]])
        cross_cov = X.T.dot(linfunc).reshape((-1,1)) * dispersion

        (pivot, 
         interval) = infer_general_target(selection_algorithm,
                                          observed_set,
                                          splitting_sampler,
                                          observed_target,
                                          cross_cov,
                                          cov_target,
                                          hypothesis=[true_target],
                                          fit_probability=probit_fit,
                                          alpha=alpha,
                                          B=1000)[:2]

        pivots.append(pivot)
        covered.append((interval[0] < true_target) * (interval[1] > true_target))
        lengths.append(interval[1] - interval[0])

        target_sd = np.sqrt(cov_target)
        quantile = ndist.ppf(1 - 0.5 * alpha)
        naive_interval = (observed_target - quantile * target_sd, observed_target + quantile * target_sd)
        naive_pivots.append((1 - ndist.cdf((observed_target - true_target) / target_sd))) # one-sided

        naive_covered.append((naive_interval[0] < true_target) * (naive_interval[1] > true_target))
        naive_lengths.append(naive_interval[1] - naive_interval[0])

    return pivots, covered, lengths, naive_pivots, naive_covered, naive_lengths


if __name__ == "__main__":
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    np.random.seed(1)

    U = np.linspace(0, 1, 101)
    P, L, coverage = [], [], []
    naive_P, naive_L, naive_coverage = [], [], []
    plt.clf()
    for i in range(500):
        p, cover, l, naive_p, naive_covered, naive_l = simulate()
        coverage.extend(cover)
        P.extend(p)
        L.extend(l)
        naive_P.extend(naive_p)
        naive_coverage.extend(naive_covered)
        naive_L.extend(naive_l)

        print("selective:", np.mean(P), np.std(P), np.mean(L) , np.mean(coverage))
        print("naive:", np.mean(naive_P), np.std(naive_P), np.mean(naive_L), np.mean(naive_coverage))
        print("len ratio selective divided by naive:", np.mean(np.array(L) / np.array(naive_L)))

        if i % 2 == 0 and i > 0:
            plt.clf()
            plt.plot(U, sm.distributions.ECDF(P)(U), 'r', label='Selective', linewidth=3)
            plt.plot([0,1], [0,1], 'k--', linewidth=2)
            plt.plot(U, sm.distributions.ECDF(naive_P)(U), 'b', label='Naive', linewidth=3)
            plt.legend()
            plt.savefig('lasso_example_CV.pdf')
